"""Ground-control-point comparison for ASP DEMs (issue #156 prototype).

Consumes a control-point GeoParquet written by `groundcontrol
<https://github.com/uw-cryo/groundcontrol>`_ — fetching, normalization, and
datum provenance live there. This module only quality-filters the points,
lands them in the DEM's frame (delegating the 3D CRS/datum/epoch transform to
``groundcontrol.crs.transform_points`` when a transform is needed), samples
the DEM, and computes the residual statistics the report figure shows.

The dependency direction is deliberate: asp_plot *optionally* imports
groundcontrol at the one call site that needs it (:func:`_transform_points`)
and fails with an actionable message when it is absent. Control parquets whose
CRS already matches the DEM (pre-transformed points, test fixtures) never
touch groundcontrol at all.
"""

import json
import logging
from datetime import datetime

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.plot as rioplot
from pyproj import CRS

from asp_plot.altimetry_source import AltimetrySource
from asp_plot.utils import Plotter, Raster, nmad, save_figure

logger = logging.getLogger(__name__)

#: NGS ``vertSource`` values considered survey-grade for DEM validation.
#: VERTCON/SCALED/POSTED heights are interpolated or map-scaled vintages.
NGS_SURVEY_GRADE_VERT_SOURCES = ("ADJUSTED", "GPS OBS", "LEVELING")

#: 3DEP ``point_type`` values usable as topographic control (BVA = bathymetry).
THREEDEP_TOPO_POINT_TYPES = ("NVA", "VVA")


def decimal_year(dt):
    """Convert a datetime to a decimal year (for the transform's ``tt``)."""
    start = datetime(dt.year, 1, 1)
    end = datetime(dt.year + 1, 1, 1)
    return dt.year + (dt - start).total_seconds() / (end - start).total_seconds()


def _transform_points(gdf, target_crs, epoch):
    """Land control points in the DEM frame via groundcontrol (lazy import)."""
    try:
        from groundcontrol.crs import transform_points
    except ImportError as e:
        raise ImportError(
            "Transforming control points into the DEM frame requires the "
            "groundcontrol package (https://github.com/uw-cryo/groundcontrol). "
            "Install it, or supply a control parquet already in the DEM CRS "
            "with ellipsoidal heights."
        ) from e
    return transform_points(gdf, target_crs, tt=epoch)


class ControlPoints:
    """Load, filter, transform, and sample ground control against an ASP DEM.

    Parameters
    ----------
    control_parquet : str
        Path to a control-point GeoParquet from ``groundcontrol-fetch``
        (or any GeoDataFrame parquet with a ``height`` column).
    dem_fn : str
        Path to the ASP DEM (heights above ellipsoid, per ASP convention).
    epoch : float or None, optional
        Decimal-year epoch of the DEM (i.e. the acquisition date), used as
        the transform's 4D time coordinate. Required when the control points
        need a frame transform; unused when they are already in the DEM CRS.
    """

    def __init__(self, control_parquet, dem_fn, epoch=None):
        self.control_parquet = control_parquet
        self.dem_fn = dem_fn
        self.epoch = epoch
        self.control_gdf = None
        self.stats = None

    def load(self):
        """Read the parquet and apply the quality filter."""
        gdf = gpd.read_parquet(self.control_parquet)
        self.control_gdf = self.filter_quality(gdf)
        logger.info(
            "Loaded %d control points (%d after quality filter) from %s",
            len(gdf),
            len(self.control_gdf),
            self.control_parquet,
        )
        return self.control_gdf

    @staticmethod
    def filter_quality(gdf):
        """Keep survey-grade points with usable heights.

        Drops rows without a height; for NGS monuments keeps only
        survey-grade ``vertSource`` vintages (parsed from the ``raw`` JSON
        column); for 3DEP checkpoints drops bathymetry (BVA); drops NGL GNSS
        stations entirely (their heights are antenna-reference, and the
        monument may sit above ground — not height-checkpoint grade). Sources
        this function does not recognize pass through untouched.
        """
        keep = gdf["height"].notna()
        if "source" in gdf.columns:
            keep &= gdf["source"] != "ngl"
            if "raw" in gdf.columns:
                ngs = gdf["source"] == "ngs"
                vert_source = gdf.loc[ngs, "raw"].map(
                    lambda s: json.loads(s).get("vertSource")
                )
                keep &= ~ngs | gdf.index.isin(
                    vert_source[vert_source.isin(NGS_SURVEY_GRADE_VERT_SOURCES)].index
                )
            if "point_type" in gdf.columns:
                threedep = gdf["source"] == "3dep"
                keep &= ~threedep | gdf["point_type"].isin(THREEDEP_TOPO_POINT_TYPES)
        return gdf[keep].copy()

    def to_dem_frame(self, gdf=None):
        """Return the control points in the DEM's 3D frame (HAE heights).

        Points already in the DEM's horizontal CRS are trusted as prepared
        (heights taken as ellipsoidal) and returned unchanged; anything else
        goes through groundcontrol's packaged 3D/4D transform, which requires
        ``self.epoch``.
        """
        if gdf is None:
            gdf = self.control_gdf if self.control_gdf is not None else self.load()
        dem_crs = CRS(Raster(self.dem_fn).ds.crs)
        if gdf.crs is not None and CRS(gdf.crs).equals(dem_crs):
            return gdf
        if self.epoch is None:
            raise ValueError(
                "Control points are not in the DEM CRS and no epoch was given. "
                "Pass epoch= (decimal year of the DEM acquisition) so the "
                "frame transform has its 4D time coordinate."
            )
        target = dem_crs.to_3d()
        # groundcontrol fail-louds on a mixed vertical_crs column (it never
        # guesses a vertical datum), so transform each uniform-datum group
        # separately; all groups land in the same DEM HAE frame. Rows with no
        # declared vertical datum are unusable and dropped.
        if "vertical_crs" in gdf.columns and gdf["vertical_crs"].nunique() > 1:
            n_undeclared = int(gdf["vertical_crs"].isna().sum())
            if n_undeclared:
                logger.warning(
                    "Dropping %d control points with no declared vertical datum",
                    n_undeclared,
                )
            parts = [
                _transform_points(group, target, self.epoch)
                for _, group in gdf.groupby("vertical_crs")
            ]
            return gpd.GeoDataFrame(
                pd.concat(parts), geometry="geometry", crs=parts[0].crs
            )
        return _transform_points(gdf, target, self.epoch)

    @staticmethod
    def _nmad_outlier_mask(dh, n_sigma):
        """Mask keeping dh within ``n_sigma`` × NMAD of the median.

        The altimetry sources filter on the classical std, which is fine at
        their point counts but breaks down for sparse control: with a single
        blunder among n points the largest |z| is (n−1)/√n, so a 3σ std
        filter can never reject anything at n ≤ 10. NMAD around the median
        has no such breakdown. Same degenerate-spread contract as
        ``AltimetrySource._std_outlier_mask``: returns ``None`` ("do not
        filter") on empty or zero/non-finite spread, and NaN rows are kept.
        """
        valid = dh.dropna().values
        if valid.size == 0:
            return None
        med = np.median(valid)
        spread = nmad(valid)
        if spread == 0 or np.isnan(spread):
            return None
        return ((dh - med).abs() <= n_sigma * spread) | dh.isna()

    def sample_dem(self, n_sigma=3):
        """Sample the DEM at the control points and compute residuals.

        Adds a ``dem_minus_control`` column (meters, DEM − control height,
        both HAE) and stores summary statistics on ``self.stats``. Outliers
        beyond ``n_sigma`` × NMAD of the median are dropped by default (see
        :meth:`_nmad_outlier_mask` for why not the altimetry sources' std
        filter); pass ``n_sigma=None`` to keep everything.
        """
        pts = self.to_dem_frame()
        dem = AltimetrySource._open_dem(self.dem_fn)
        sampled, pts = AltimetrySource._interp_dem_at_points(dem, pts)
        pts["dem_minus_control"] = sampled - pts["height"].values

        if n_sigma is not None:
            mask = self._nmad_outlier_mask(pts["dem_minus_control"], n_sigma)
            if mask is not None:
                n_out = int((~mask).sum())
                if n_out:
                    logger.info("Outlier filter (%dσ NMAD): removed %d", n_sigma, n_out)
                pts = pts[mask]

        dh = pts["dem_minus_control"].dropna()
        self.stats = {
            "n": int(len(dh)),
            "median": float(np.median(dh)) if len(dh) else np.nan,
            "nmad": float(nmad(dh)) if len(dh) else np.nan,
            "mean": float(np.mean(dh)) if len(dh) else np.nan,
            "std": float(np.std(dh)) if len(dh) else np.nan,
        }
        self.control_gdf = pts
        return pts


class ControlPointsPlotter(Plotter):
    """Render the ground-control residual figure for the report."""

    def __init__(self, control_points, **kwargs):
        super().__init__(**kwargs)
        self.cp = control_points

    def plot_control_dh(self, clim=None, save_dir=None, fig_fn=None):
        """Two panels: dh points over the DEM hillshade, and a histogram."""
        pts = self.cp.control_gdf
        if pts is None or "dem_minus_control" not in pts.columns:
            pts = self.cp.sample_dem()
        gdf = pts.dropna(subset=["dem_minus_control"])
        stats = self.cp.stats

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=220)
        if gdf.empty:
            for ax in axes:
                self.plot_missing(ax, "No control points\nover valid DEM area")
        else:
            dh = gdf["dem_minus_control"]
            if clim is None:
                abs_max = float(np.nanmax(np.abs(dh.values)))
                clim = (-abs_max, abs_max)

            dem_raster = Raster(self.cp.dem_fn, downsample=4)
            hs = dem_raster.hillshade()
            extent = rioplot.plotting_extent(
                dem_raster.ds, transform=dem_raster.transform
            )
            ax = axes[0]
            ax.imshow(hs, cmap="gray", extent=extent, alpha=0.7, interpolation="none")
            gdf.plot(
                ax=ax,
                column="dem_minus_control",
                cmap="RdBu",
                vmin=clim[0],
                vmax=clim[1],
                markersize=12,
                edgecolor="k",
                linewidth=0.3,
                legend=True,
                legend_kwds={"label": "DEM - control (m)\n[±|max|]"},
            )
            ax.set_title("Control point residuals", size=9)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = axes[1]
            abs_max = max(abs(dh.min()), abs(dh.max()))
            bins = np.linspace(-abs_max, abs_max, min(65, max(9, len(dh))))
            ax.hist(dh.values, bins=bins, color="steelblue", alpha=0.8)
            ax.axvline(0, color="k", lw=0.5)
            stats_text = (
                f"n={stats['n']}\n"
                f"Median={stats['median']:+.2f} m\n"
                f"NMAD={stats['nmad']:.2f} m"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
            )
            ax.set_xlabel("DEM - control (m)")
            ax.set_ylabel("Count")
            ax.set_title("Residual distribution", size=9)

        fig.suptitle(self.title or "Ground control vs DEM", size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
