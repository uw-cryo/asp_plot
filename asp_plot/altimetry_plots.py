"""Altimetry plotting layer.

All figure rendering for :class:`asp_plot.altimetry.Altimetry`, split out from
the data-source code so plot methods operate on **already-prepared**
dataframes. The coordinating ``Altimetry`` instance computes the dh columns
(``atl06sr_to_dem_dh`` / ``planetary_to_dem_dh``) and resolves the best track
*before* calling into this module, so no plot body triggers a SlideRule
request or re-samples a DEM for differencing. Reading a DEM purely to draw a
hillshade backdrop is a rendering concern and stays here.

The plotter reads scalar context (``dem_fn``, ``aligned_dem_fn``, the request
time-range label) from the coordinator passed at construction; the
dataframes themselves arrive as method arguments.
"""

import logging

import contextily as ctx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from rasterio import plot as rioplot

from asp_plot.bodies import BODIES
from asp_plot.utils import ColorBar, Raster
from asp_plot.utils import nmad as _nmad
from asp_plot.utils import save_figure

logger = logging.getLogger(__name__)

WORLDCOVER_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse",
    70: "Snow/ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss/lichen",
}


class AltimetryPlotter:
    """Render ICESat-2 and planetary altimetry figures from prepared data.

    Parameters
    ----------
    alt : Altimetry
        The coordinating :class:`asp_plot.altimetry.Altimetry` instance.
        Used only to read scalar context (``dem_fn``, ``aligned_dem_fn``,
        and the ICESat-2 request time-range label). All dataframes are
        passed explicitly to each plot method.
    """

    def __init__(self, alt):
        self.alt = alt

    @property
    def _time_range_label(self):
        """ICESat-2 request time-range label for plot titles (or empty)."""
        return self.alt.icesat2._time_range_label

    # ------------------------------------------------------------------ #
    #  Planetary altimetry plots                                         #
    # ------------------------------------------------------------------ #

    def mapview_plot_planetary_to_dem(
        self,
        planetary_points,
        clim=None,
        save_dir=None,
        fig_fn=None,
        title=None,
        plot_aligned=False,
    ):
        """Map view of planetary altimetry vs DEM height differences.

        Plots the DEM hillshade as background with altimetry dh points
        overlaid using a divergent colourmap. When ``plot_aligned=True``
        and ``self.alt.aligned_dem_fn`` is set, renders pre/post panels side
        by side.

        Parameters
        ----------
        planetary_points : geopandas.GeoDataFrame
            Prepared planetary points with ``altimetry_minus_dem`` (and,
            when available, ``altimetry_minus_aligned_dem``) columns.
        clim : tuple or None, optional
            Colour limits ``(min, max)`` for dh. Default auto (symmetric
            ±|max| around zero).
        save_dir : str or None, optional
            Directory to save figure.
        fig_fn : str or None, optional
            Filename for saved figure.
        title : str or None, optional
            Custom plot title. Auto-detected if None.
        plot_aligned : bool, optional
            Add a second panel showing dh against the aligned DEM.
            Requires that pc_align has been run successfully.
        """
        from asp_plot.utils import detect_planetary_body

        if planetary_points is None or planetary_points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return

        body = detect_planetary_body(self.alt.dem_fn)
        instrument = BODIES[body].altimetry_instrument
        if title is None:
            title = f"{instrument} vs DEM"

        show_aligned = (
            plot_aligned
            and self.alt.aligned_dem_fn
            and "altimetry_minus_aligned_dem" in planetary_points.columns
        )

        # Generate hillshade — use the aligned DEM as the backdrop when
        # available so the post-alignment panel matches the dh sample.
        backdrop_dem = self.alt.aligned_dem_fn if show_aligned else self.alt.dem_fn
        dem_raster = Raster(backdrop_dem, downsample=4)
        hs = dem_raster.hillshade()
        extent = rioplot.plotting_extent(dem_raster.ds, transform=dem_raster.transform)
        dem_crs = dem_raster.ds.crs

        # Build the symmetric ±|max| color limits across all visible
        # panels so they are directly comparable.
        gdf_unaligned = planetary_points.dropna(subset=["altimetry_minus_dem"])
        if gdf_unaligned.empty:
            logger.warning("No valid dh values for map view.")
            return
        dh_arrays = [gdf_unaligned["altimetry_minus_dem"].values]
        if show_aligned:
            dh_arrays.append(
                planetary_points["altimetry_minus_aligned_dem"].dropna().values
            )

        if clim is None:
            abs_max = max(np.nanmax(np.abs(a)) for a in dh_arrays if a.size)
            clim = (-abs_max, abs_max)

        ncols = 2 if show_aligned else 1
        fig, axes = plt.subplots(
            1, ncols, figsize=(8 * ncols, 6), dpi=220, squeeze=False
        )
        axes = axes[0]

        panels = [("altimetry_minus_dem", "ASP DEM")]
        if show_aligned:
            panels.append(("altimetry_minus_aligned_dem", "Aligned DEM"))

        for ax, (column, label) in zip(axes, panels):
            gdf = planetary_points.dropna(subset=[column])
            dh = gdf[column]
            n = len(dh)
            med = np.nanmedian(dh.values)
            nmad = _nmad(dh.values)

            ax.imshow(hs, cmap="gray", extent=extent, alpha=0.7, interpolation="none")
            cbar_label = f"{instrument} - {label} (m)\n[±|max|]"
            gdf.to_crs(dem_crs).plot(
                ax=ax,
                column=column,
                cmap="RdBu",
                vmin=clim[0],
                vmax=clim[1],
                markersize=2,
                legend=True,
                legend_kwds={"label": cbar_label},
            )
            stats_text = f"n={n}\nMedian={med:+.2f} m\nNMAD={nmad:.2f} m"
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
            ax.set_title(label, size=9)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(title, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def histogram_planetary_to_dem(
        self,
        planetary_points,
        save_dir=None,
        fig_fn=None,
        title=None,
        plot_aligned=False,
    ):
        """Histogram of planetary altimetry vs DEM height differences.

        Parameters
        ----------
        planetary_points : geopandas.GeoDataFrame
            Prepared planetary points with ``altimetry_minus_dem`` (and,
            when available, ``altimetry_minus_aligned_dem``) columns.
        save_dir : str or None, optional
            Directory to save figure.
        fig_fn : str or None, optional
            Filename for saved figure.
        title : str or None, optional
            Custom plot title. Auto-detected if None.
        plot_aligned : bool, optional
            Overlay the post-alignment dh distribution on the same axes.
            Requires that pc_align has been run successfully.
        """
        from asp_plot.utils import detect_planetary_body

        if planetary_points is None or planetary_points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return

        dh = planetary_points["altimetry_minus_dem"].dropna()
        if dh.empty:
            logger.warning("No valid dh values for histogram.")
            return

        body = detect_planetary_body(self.alt.dem_fn)
        instrument = BODIES[body].altimetry_instrument
        if title is None:
            title = f"{instrument} vs ASP DEM"

        show_aligned = (
            plot_aligned
            and self.alt.aligned_dem_fn
            and "altimetry_minus_aligned_dem" in planetary_points.columns
        )
        if show_aligned:
            dh_aligned = planetary_points["altimetry_minus_aligned_dem"].dropna()

        # Shared bin edges and xlim across both distributions
        if show_aligned:
            abs_max = max(
                abs(dh.min()),
                abs(dh.max()),
                abs(dh_aligned.min()),
                abs(dh_aligned.max()),
            )
        else:
            abs_max = max(abs(dh.min()), abs(dh.max()))
        bins = np.linspace(-abs_max, abs_max, 129)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=220)
        ax.hist(
            dh.values,
            bins=bins,
            alpha=0.6,
            color="steelblue",
            label=f"DEM (n={len(dh)})",
        )
        if show_aligned:
            ax.hist(
                dh_aligned.values,
                bins=bins,
                alpha=0.6,
                color="orange",
                label=f"Aligned DEM (n={len(dh_aligned)})",
            )

        ax.set_xlim(-abs_max, abs_max)

        if show_aligned:
            med0 = np.nanmedian(dh.values)
            nmad0 = _nmad(dh.values)
            med1 = np.nanmedian(dh_aligned.values)
            nmad1 = _nmad(dh_aligned.values)
            stats_text = (
                f"DEM:         Med={med0:+.2f}  NMAD={nmad0:.2f} m\n"
                f"Aligned DEM: Med={med1:+.2f}  NMAD={nmad1:.2f} m"
            )
        else:
            n = len(dh)
            med = np.nanmedian(dh.values)
            nmad = _nmad(dh.values)
            stats_text = f"n={n}\nMedian={med:+.2f} m\nNMAD={nmad:.2f} m"
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

        ax.set_xlabel(f"{instrument} - DEM (m) [±|max|]")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(title, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    # ------------------------------------------------------------------ #
    #  ICESat-2 plots                                                    #
    # ------------------------------------------------------------------ #

    def plot_atl06sr_time_stamps(
        self,
        filtered,
        key="all",
        title="ICESat-2 ATL06-SR Time Stamps",
        cmap="inferno",
        map_crs="EPSG:4326",
        figsize=(15, 10),
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        """
        Plot ATL06-SR data for different temporal filters.

        Creates a 2x2 grid of plots showing ATL06-SR data for different
        temporal filters (unfiltered, 15-day, 45-day, and seasonal)
        colored by height.

        Parameters
        ----------
        filtered : dict
            The ``atl06sr_processing_levels_filtered`` mapping of
            processing-level key -> GeoDataFrame.
        key : str, optional
            Base processing level to plot, default is "all"
        title : str, optional
            Plot title, default is "ICESat-2 ATL06-SR Time Stamps"
        cmap : str, optional
            Matplotlib colormap for elevation, default is "inferno"
        map_crs : str, optional
            Coordinate reference system for mapping, default is "EPSG:4326"
        figsize : tuple, optional
            Figure size as (width, height), default is (15, 10)
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        **ctx_kwargs : dict, optional
            Additional arguments for contextily basemap

        Notes
        -----
        This method requires the filtered data to have been created using
        the predefined_temporal_filter_atl06sr method for the temporal
        variations to be available.
        """
        time_stamps = ["", "_15_day_pad", "_45_day_pad", "_seasonal"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        x_bounds = []
        y_bounds = []
        # CRS to reproject every panel into. This is independent of whether a
        # basemap is drawn: a basemap is only fetched when the caller passes
        # contextily kwargs (e.g. source=...). Previously this CRS was injected
        # into ``ctx_kwargs``, which made the ``if ctx_kwargs`` basemap guard
        # below always true and forced a tile download on every call (a network
        # hang in offline/CI runs).
        if map_crs:
            crs = map_crs
        elif ctx_kwargs:
            crs = ctx_kwargs.get("crs", "EPSG:4326")
        else:
            crs = "EPSG:4326"
        for ax, time_stamp in zip(axes, time_stamps):
            key_to_plot = f"{key}{time_stamp}"

            if key_to_plot not in filtered.keys():
                ax.text(
                    0.5,
                    0.5,
                    f"No points found for {key_to_plot}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            atl06sr = filtered[key_to_plot]
            atl06sr_sorted = atl06sr.sort_values(by="h_mean").to_crs(crs)
            bounds = atl06sr_sorted.total_bounds
            x_bounds.extend([bounds[0], bounds[2]])
            y_bounds.extend([bounds[1], bounds[3]])

            cb = ColorBar(perc_range=(2, 98))
            cb.get_clim(atl06sr_sorted["h_mean"])
            norm = cb.get_norm(lognorm=False)

            atl06sr_sorted.plot(
                ax=ax,
                column="h_mean",
                cmap=cmap,
                norm=norm,
                s=1,
                legend=True,
                legend_kwds={"label": "Height above datum (m)"},
            )

            ax.set_title(f"{key_to_plot} (n={atl06sr.shape[0]})", size=12)

        # 5% padding
        padding = 0.05
        x_range = max(x_bounds) - min(x_bounds)
        y_range = max(y_bounds) - min(y_bounds)
        for ax, time_stamp in zip(axes, time_stamps):
            key_to_plot = f"{key}{time_stamp}"

            if key_to_plot not in filtered.keys():
                continue

            ax.set_xlim(
                min(x_bounds) - padding * x_range, max(x_bounds) + padding * x_range
            )
            ax.set_ylim(
                min(y_bounds) - padding * y_range, max(y_bounds) + padding * y_range
            )
            if ctx_kwargs:
                ctx_kwargs.setdefault("crs", crs)
                ctx.add_basemap(ax=ax, **ctx_kwargs)

        suptitle = f"{title}"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=14)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_atl06sr(
        self,
        atl06sr,
        key="all",
        plot_beams=False,
        plot_dem=False,
        column_name="h_mean",
        cbar_label="Height above datum (m)",
        title="ICESat-2 ATL06-SR",
        clim=None,
        symm_clim=False,
        cmap="inferno",
        map_crs="EPSG:4326",
        figsize=(6, 4),
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        """
        Plot ATL06-SR data on a map with customizable options.

        Creates a map view of ATL06-SR data with options to color by various
        attributes, highlight different laser beams, overlay on the DEM,
        and add contextual basemaps.

        Parameters
        ----------
        atl06sr : geopandas.GeoDataFrame
            Prepared ATL06-SR points to plot (one processing level).
        key : str, optional
            Processing level label for the title, default is "all"
        plot_beams : bool, optional
            Whether to color points by ICESat-2 beam, default is False
        plot_dem : bool, optional
            Whether to plot the DEM as a background, default is False
        column_name : str, optional
            Column to use for point coloring, default is "h_mean"
        cbar_label : str, optional
            Colorbar label, default is "Height above datum (m)"
        title : str, optional
            Plot title, default is "ICESat-2 ATL06-SR"
        clim : tuple or None, optional
            Color limits as (min, max), default is None (auto)
        symm_clim : bool, optional
            Whether to use symmetric color limits, default is False
        cmap : str, optional
            Matplotlib colormap, default is "inferno"
        map_crs : str, optional
            Coordinate reference system for mapping, default is "EPSG:4326"
        figsize : tuple, optional
            Figure size as (width, height), default is (6, 4)
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        **ctx_kwargs : dict, optional
            Additional arguments for contextily basemap

        Notes
        -----
        When plot_beams is True, points are colored by ICESat-2 laser spot
        number, with strong beams (1, 3, 5) in darker colors and
        weak beams (2, 4, 6) in lighter colors.
        """
        atl06sr_sorted = atl06sr.sort_values(by=column_name).to_crs(map_crs)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=220)

        if plot_dem:
            ctx_kwargs = {}
            # We downsample to speed plotting. This is not carried over into any analysis.
            dem_downsampled = Raster(self.alt.dem_fn, downsample=10)
            cb = ColorBar(perc_range=(2, 98))
            cb.get_clim(dem_downsampled.data)
            # Plot using rasterio's show function
            rioplot.show(
                dem_downsampled.data,
                transform=dem_downsampled.transform,
                ax=ax,
                cmap="inferno",
                vmin=cb.clim[0],
                vmax=cb.clim[1],
                alpha=1,
            )
            ax.set_title(None)

        # TODO: Implement optional hillshade plotting

        if plot_beams:
            color_dict = {
                1: "red",
                2: "lightpink",
                3: "blue",
                4: "lightblue",
                5: "green",
                6: "lightgreen",
            }
            patches = [mpatches.Patch(color=v, label=k) for k, v in color_dict.items()]
            atl06sr_sorted.plot(
                ax=ax,
                markersize=1,
                color=atl06sr_sorted["spot"].map(color_dict).values,
            )
            ax.legend(
                handles=patches, title="laser spot\n(strong=1,3,5)", loc="upper left"
            )
        else:
            if plot_dem:
                cb.symm = symm_clim
            else:
                cb = ColorBar(perc_range=(2, 98), symm=symm_clim)
                if clim is None:
                    cb.get_clim(atl06sr_sorted[column_name])
                else:
                    cb.clim = clim

            norm = cb.get_norm(lognorm=False)

            atl06sr_sorted.plot(
                ax=ax,
                column=column_name,
                cmap=cmap,
                norm=norm,
                s=1,
                legend=True,
                legend_kwds={"label": cbar_label},
            )

        if ctx_kwargs:
            ctx.add_basemap(ax=ax, **ctx_kwargs)

        ax.set_xticks([])
        ax.set_yticks([])

        suptitle = f"{title}\n{key} (n={atl06sr.shape[0]})"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def mapview_plot_atl06sr_to_dem(
        self,
        atl06sr,
        key="all",
        clim=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
        map_crs=None,
        **ctx_kwargs,
    ):
        """
        Plot height differences between ATL06-SR data and DEMs.

        Creates a map visualization of the height differences between
        ICESat-2 ATL06-SR data and either the original or aligned DEM.

        Parameters
        ----------
        atl06sr : geopandas.GeoDataFrame
            Prepared ATL06-SR points with the dh column already computed.
        key : str, optional
            Processing level label, default is "all"
        clim : tuple or None, optional
            Color limits as (min, max), default is None (auto)
        plot_aligned : bool, optional
            Whether to plot differences with aligned DEM, default is False
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        map_crs : str or None, optional
            Coordinate reference system for mapping, default is None
            (use DEM's CRS)
        **ctx_kwargs : dict, optional
            Additional arguments for contextily basemap

        Notes
        -----
        The plot uses a divergent colormap (RdBu) to highlight
        positive and negative differences.
        """
        if plot_aligned:
            column_name = "icesat_minus_aligned_dem"
            if not self.alt.aligned_dem_fn:
                print("\nAligned DEM not found.\n")
                return
        else:
            column_name = "icesat_minus_dem"

        if clim is None:
            # Symmetric ±3σ centered on 0 (data has already been filtered
            # to within 3σ by atl06sr_to_dem_dh, so min/max of the filtered
            # values ≈ ±3σ from the mean)
            dh = atl06sr[column_name].dropna()
            if not dh.empty:
                abs_max = max(abs(dh.min()), abs(dh.max()))
                clim = (-abs_max, abs_max)
            cbar_label = "ICESat-2 minus DEM (m)\n[±3σ]"
        else:
            cbar_label = "ICESat-2 minus DEM (m)"

        if not map_crs:
            dem = rioxarray.open_rasterio(self.alt.dem_fn, masked=True).squeeze()
            epsg = dem.rio.crs.to_epsg()
            map_crs = f"EPSG:{epsg}"

        self.plot_atl06sr(
            atl06sr,
            key=key,
            column_name=column_name,
            cbar_label=cbar_label,
            clim=clim,
            symm_clim=False,
            cmap="RdBu",
            map_crs=map_crs,
            save_dir=save_dir,
            fig_fn=fig_fn,
            **ctx_kwargs,
        )

    def histogram(
        self,
        atl06sr,
        key="all",
        title="Histogram",
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot histograms of height differences between ATL06-SR data and DEMs.

        Creates histograms of the height differences between ICESat-2 ATL06-SR
        data and DEMs, with statistics including median and normalized median
        absolute deviation (NMAD).

        Parameters
        ----------
        atl06sr : geopandas.GeoDataFrame
            Prepared ATL06-SR points with the dh column already computed.
        key : str, optional
            Processing level label, default is "all"
        title : str, optional
            Plot title, default is "Histogram"
        plot_aligned : bool, optional
            Whether to include differences with aligned DEM, default is False
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None

        Notes
        -----
        NMAD is a robust measure of dispersion that is less sensitive
        to outliers than standard deviation, calculated as
        1.4826 * median(abs(x - median(x))).
        """
        column_names = ["icesat_minus_dem"]
        if plot_aligned:
            column_names.append("icesat_minus_aligned_dem")
            if not self.alt.aligned_dem_fn:
                print("\nAligned DEM not found.\n")
                return

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=220)

        abs_max = 0.0
        for column_name in column_names:
            col = atl06sr[column_name].dropna()
            med = col.quantile(0.50)
            nmad = _nmad(col.values)

            abs_max = max(abs_max, abs(col.min()), abs(col.max()))
            plot_kwargs = {"bins": 128, "alpha": 0.5}
            atl06sr.hist(
                ax=ax,
                column=column_name,
                label=f"{column_name}, Median={med:0.2f}, NMAD={nmad:0.2f}",
                **plot_kwargs,
            )

        # Symmetric ±3σ centered on 0 (data has already been 3σ-filtered
        # in atl06sr_to_dem_dh; stats displayed are median/NMAD over the
        # filtered data)
        ax.set_xlim(-abs_max, abs_max)
        ax.legend()
        ax.set_title(None)
        ax.set_xlabel("ICESat-2 - DEM (m) [±3σ]")

        suptitle = f"{title}\n{key} (n={atl06sr.shape[0]})"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=10)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def histogram_by_landcover(
        self,
        atl06sr,
        key="all",
        top_n=4,
        title="ICESat-2 ATL06-SR vs DEM",
        xlim=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot histogram of dh with per-landcover-class statistics.

        Creates a histogram of the height differences between ICESat-2
        ATL06-SR data and the DEM, with a text annotation showing overall
        and per-landcover-class statistics (count, median, NMAD).

        When ``plot_aligned=True`` and an aligned DEM is available,
        overlays the pre- and post-alignment distributions and renders two
        vertically stacked stats text boxes whose outline colors match the
        bar colors (color serves as the legend).

        Parameters
        ----------
        atl06sr : geopandas.GeoDataFrame
            Prepared ATL06-SR points with the dh column already computed.
        key : str, optional
            Processing level label, default is "all"
        top_n : int, optional
            Number of top landcover classes to report, default is 4
        title : str, optional
            Plot title, default is "ICESat-2 ATL06-SR vs DEM"
        xlim : tuple or None, optional
            Symmetric x-axis limits as (min, max). If None, uses ±3σ
            range (data is already 3σ-filtered in atl06sr_to_dem_dh).
        plot_aligned : bool, optional
            Whether to overlay the aligned-DEM distribution alongside the
            unaligned one. Requires ``self.alt.aligned_dem_fn`` and the
            ``icesat_minus_aligned_dem`` column. Default is False.
        save_dir : str or None, optional
            Directory to save figure, default is None
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        """
        distributions = [("icesat_minus_dem", "steelblue", "DEM")]
        if plot_aligned:
            if not self.alt.aligned_dem_fn:
                logger.warning(
                    "\nplot_aligned=True but no aligned DEM is available; "
                    "plotting unaligned distribution only.\n"
                )
            elif "icesat_minus_aligned_dem" not in atl06sr.columns:
                logger.warning(
                    "\n'icesat_minus_aligned_dem' column missing; call "
                    "atl06sr_to_dem_dh() after setting aligned_dem_fn.\n"
                )
            else:
                distributions.append(
                    ("icesat_minus_aligned_dem", "darkorange", "Aligned DEM")
                )

        # Preserve the pre-existing "All" header label when there is only
        # one distribution (no plot_aligned overlay). This keeps reports
        # generated with plot_aligned=False textually identical to prior
        # versions.
        if len(distributions) == 1:
            col, color, _ = distributions[0]
            distributions = [(col, color, "All")]

        dh_series = [
            (col, color, label, atl06sr[col].dropna())
            for col, color, label in distributions
        ]
        dh_series = [t for t in dh_series if not t[3].empty]
        if not dh_series:
            logger.warning(f"\nNo valid dh values for key: {key}\n")
            return

        if xlim is not None:
            xmin, xmax = xlim
            xlabel_note = ""
        else:
            abs_max = max(
                max(abs(dv.min()), abs(dv.max())) for _, _, _, dv in dh_series
            )
            xmin, xmax = -abs_max, abs_max
            xlabel_note = " [±3σ]"

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=220)

        # Shared bin edges so pre- and post-alignment bars line up exactly
        # (default bins=128 recomputes edges per call, which leaves the
        # two distributions with incompatible binning).
        shared_bins = np.linspace(xmin, xmax, 129)
        for _, color, _, dv in dh_series:
            ax.hist(dv.values, bins=shared_bins, alpha=0.55, color=color)
        ax.set_xlim(xmin, xmax)

        stats_blocks = [
            (
                self._build_landcover_stats_text(atl06sr, col, label, top_n=top_n),
                color,
            )
            for col, color, label, _ in dh_series
        ]

        # Stack text boxes vertically at top-left. Approximate each box's
        # height from its line count so the second box doesn't overlap the
        # first. Empirical constants tuned for fontsize=8 monospace in an
        # 8x5 inch axes.
        line_h_axes = 0.030
        pad_axes = 0.04
        gap_axes = 0.02
        box_y = 0.98
        for text, color in stats_blocks:
            ax.text(
                0.02,
                box_y,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=8,
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.5,
                    alpha=0.9,
                ),
            )
            nlines = text.count("\n") + 1
            box_y -= nlines * line_h_axes + pad_axes + gap_axes

        ax.set_xlabel(f"ICESat-2 - DEM (m){xlabel_note}")
        ax.set_ylabel("Count")
        overall_n = len(dh_series[0][3])
        suptitle = f"{title}\n{key} (n={overall_n})"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def _build_landcover_stats_text(self, atl06sr, dh_col, label, top_n=4):
        """Build the multi-line stats text for one dh column.

        Includes an "all" header line (n, Med, NMAD) and up to ``top_n``
        per-landcover-class rows, each requiring at least 10 points.
        """
        dv = atl06sr[dh_col].dropna()
        overall_n = len(dv)
        overall_med = np.nanmedian(dv.values)
        overall_nmad = _nmad(dv.values)
        lines = [
            f"{label}: n={overall_n}, Med={overall_med:+.2f} m, NMAD={overall_nmad:.2f} m"
        ]

        wc_col = "esa_worldcover.value"
        if wc_col in atl06sr.columns:
            valid_wc = atl06sr.dropna(subset=[dh_col, wc_col])
            if not valid_wc.empty:
                valid_wc = valid_wc.copy()
                valid_wc["lc_name"] = (
                    valid_wc[wc_col].map(WORLDCOVER_NAMES).fillna("Unknown")
                )
                class_stats = []
                for name, group in valid_wc.groupby("lc_name")[dh_col]:
                    if len(group) >= 10:
                        class_stats.append(
                            {
                                "name": name,
                                "n": len(group),
                                "med": np.nanmedian(group.values),
                                "nmad": _nmad(group.values),
                            }
                        )
                class_stats.sort(key=lambda x: x["n"], reverse=True)
                class_stats = class_stats[:top_n]
                if class_stats:
                    lines.append("─" * 35)
                    for cs in class_stats:
                        lines.append(
                            f"{cs['name']}: n={cs['n']}, Med={cs['med']:+.2f}, "
                            f"NMAD={cs['nmad']:.2f}"
                        )
        return "\n".join(lines)

    def _plot_hillshade_map(self, ax, track, seg_info=None):
        """
        Plot DEM hillshade with track overlay on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        track : GeoDataFrame
        seg_info : dict or None
            Output from ``Icesat2Source._find_best_worst_segments``.
        """
        # Use brighter colors on the map view for visibility against
        # the hillshade+terrain background
        seg_best_color = "#00e5ff"  # bright cyan-blue
        seg_worst_color = "#ff1744"  # bright red
        try:
            raster = Raster(self.alt.dem_fn, downsample=5)
            dem_data, dem_extent = raster.read_array(extent=True)
            epsg = raster.get_epsg_code()

            from matplotlib.colors import LightSource

            ls = LightSource(azdeg=315, altdeg=45)
            fill_val = np.nanmedian(np.asarray(dem_data))
            dem_filled = np.asarray(np.ma.filled(dem_data, fill_val))
            hs = ls.hillshade(dem_filled)

            ax.imshow(
                hs, extent=dem_extent, cmap="gray", origin="upper", aspect="equal"
            )
            ax.imshow(
                dem_data,
                extent=dem_extent,
                cmap="terrain",
                alpha=0.4,
                origin="upper",
                aspect="equal",
            )

            track_proj = track.to_crs(f"EPSG:{epsg}")
            ax.plot(
                track_proj.geometry.x,
                track_proj.geometry.y,
                color="black",
                linewidth=2,
                label="Track",
                zorder=5,
            )

            if seg_info is not None:
                for mask_key, color, label in [
                    ("seg_best_mask", seg_best_color, "Better agreement"),
                    ("seg_worst_mask", seg_worst_color, "Worse agreement"),
                ]:
                    seg_proj = track_proj.loc[seg_info[mask_key]]
                    if not seg_proj.empty:
                        ax.plot(
                            seg_proj.geometry.x,
                            seg_proj.geometry.y,
                            color=color,
                            linewidth=4,
                            label=label,
                            zorder=6,
                        )

            track_bounds = track_proj.total_bounds
            dx = track_bounds[2] - track_bounds[0]
            dy = track_bounds[3] - track_bounds[1]
            pad = max(dx, dy) * 0.2
            ax.set_xlim(track_bounds[0] - pad, track_bounds[2] + pad)
            ax.set_ylim(track_bounds[1] - pad, track_bounds[3] + pad)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(fontsize=8, loc="upper left")
        except Exception:
            logger.warning("Could not generate map view for profile plot.")
            ax.text(
                0.5,
                0.5,
                "Map view unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])

    @staticmethod
    def _add_segment_spans(ax, seg_info):
        """Shade the best/worst 1 km segment extents on a profile axis."""
        if seg_info is None:
            return
        for start, end, color in [
            (seg_info["seg_best_start_km"], seg_info["seg_best_end_km"], "tab:blue"),
            (seg_info["seg_worst_start_km"], seg_info["seg_worst_end_km"], "tab:red"),
        ]:
            ax.axvspan(start, end, alpha=0.15, color=color, zorder=0)

    def _profile_elevation_panel(self, ax_elev, track, dist, seg_info, plot_aligned):
        """Top profile panel: absolute DEM/COP30/aligned/ICESat-2 elevations."""
        valid_dem = track["dem_height"].dropna()
        if not valid_dem.empty:
            ax_elev.plot(
                dist.loc[valid_dem.index],
                valid_dem,
                color="gray",
                linewidth=1,
                label="ASP DEM",
                zorder=1,
            )

        # COP30 sampled height (if available from SlideRule)
        cop30_col = "cop30.value"
        if cop30_col in track.columns:
            valid_cop30 = track[cop30_col].dropna()
            if not valid_cop30.empty:
                ax_elev.scatter(
                    dist.loc[valid_cop30.index],
                    valid_cop30,
                    color="darkgoldenrod",
                    s=4,
                    alpha=0.6,
                    label="COP30",
                    zorder=2,
                )

        if plot_aligned and self.alt.aligned_dem_fn:
            if "aligned_dem_height" in track.columns:
                valid_aligned = track["aligned_dem_height"].dropna()
                if not valid_aligned.empty:
                    ax_elev.plot(
                        dist.loc[valid_aligned.index],
                        valid_aligned,
                        color="orange",
                        linewidth=1,
                        label="Aligned DEM",
                        zorder=3,
                    )

        ax_elev.scatter(
            dist,
            track["h_mean"],
            color="steelblue",
            s=8,
            label="ICESat-2 ATL06-SR",
            zorder=4,
        )

        self._add_segment_spans(ax_elev, seg_info)

        ax_elev.set_ylabel("Elevation (m HAE)")
        ax_elev.legend(fontsize=8, loc="upper left")
        ax_elev.grid(True, color="lightgray", linewidth=0.5, alpha=0.7)
        ax_elev.set_axisbelow(True)
        plt.setp(ax_elev.get_xticklabels(), visible=False)

    def _profile_dh_panel(self, ax_dh, track, dist, dh_vals, seg_info, plot_aligned):
        """Bottom profile panel: the dh scatter (post-alignment when active)."""
        # When plot_aligned is active, show the *post*-alignment dh and
        # recompute Med/NMAD against the aligned DEM so the bottom panel
        # reflects what the aligned-DEM page is actually evaluating.
        use_aligned_dh = (
            plot_aligned
            and self.alt.aligned_dem_fn
            and "icesat_minus_aligned_dem" in track.columns
        )
        if use_aligned_dh:
            dh_plot = track["icesat_minus_aligned_dem"].dropna()
            dh_label_suffix = " (Aligned DEM)"
        else:
            dh_plot = dh_vals
            dh_label_suffix = ""

        if not dh_plot.empty:
            med = np.nanmedian(dh_plot.values)
            nmad_val = _nmad(dh_plot.values)
            ax_dh.scatter(
                dist.loc[dh_plot.index],
                dh_plot,
                color="gray",
                s=4,
                alpha=0.6,
                zorder=2,
                label=f"Med={med:+.2f} m, NMAD={nmad_val:.2f} m{dh_label_suffix}",
            )
            ax_dh.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)

        self._add_segment_spans(ax_dh, seg_info)

        ax_dh.set_ylabel("ICESat-2 − DEM (m)")
        ax_dh.set_xlabel("Along-track distance (km)")
        ax_dh.legend(fontsize=8, loc="upper left")
        ax_dh.grid(True, color="lightgray", linewidth=0.5, alpha=0.7)
        ax_dh.set_axisbelow(True)

    def plot_atl06sr_dem_profile(
        self,
        resolved,
        seg_info,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot elevation profile comparing ICESat-2 and DEM along the best track.

        Creates a 2×2 figure with the profile stack on the left and a map
        view spanning the full height on the right:
        - Top-left: Absolute elevation profile (DEM, COP30, ICESat-2)
        - Bottom-left: Height difference profile (ICESat-2 minus DEM)
          (shares x-axis with top-left, no vertical space between them)
        - Right column: DEM hillshade map with the full track and segment
          extents, spanning the full vertical height

        Parameters
        ----------
        resolved : tuple
            The ``(track, rgt, cycle, spot, track_count, track_date, dist,
            dh_vals)`` tuple from ``Icesat2Source._resolve_best_track``.
        seg_info : dict or None
            Best/worst segment extents from
            ``Icesat2Source._find_best_worst_segments``.
        plot_aligned : bool, optional
            Whether to also plot the aligned DEM profile, default is False
        save_dir : str or None, optional
            Directory to save figure, default is None
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        """
        track, rgt, cycle, spot, track_count, track_date, dist, dh_vals = resolved

        # --- Figure layout: left column = stacked elevation/dh (no gap),
        # right column = map spanning both rows ---
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[2, 1.2],
            width_ratios=[2, 1],
            hspace=0.0,
            wspace=0.1,
        )
        ax_elev = fig.add_subplot(gs[0, 0])
        ax_dh = fig.add_subplot(gs[1, 0], sharex=ax_elev)
        ax_map = fig.add_subplot(gs[:, 1])

        self._profile_elevation_panel(ax_elev, track, dist, seg_info, plot_aligned)
        self._profile_dh_panel(ax_dh, track, dist, dh_vals, seg_info, plot_aligned)

        # =================== Row 3: Map view ====================
        self._plot_hillshade_map(ax_map, track, seg_info)

        # Title
        title_str = f"RGT {rgt}, Cycle {cycle}, Spot {spot} ({track_date})"
        if track_count:
            title_str += f" — n={track_count}"
        if self._time_range_label:
            title_str += f"\n{self._time_range_label}"
        fig.suptitle(title_str, size=10)

        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.07, right=0.97)
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def _draw_segment_panel(
        self, ax_seg, seg, color, label_prefix, show_aligned, dh_col
    ):
        """Render one (better/worse agreement) 1 km segment panel.

        Plots DEM / aligned-DEM / COP30 / ICESat-2 elevations within the
        segment and titles it with the per-segment Median/NMAD.
        """
        seg_dem = seg["dem_height"].dropna()
        seg_h = seg["h_mean"].dropna()
        if not seg_dem.empty:
            seg_dem_dist = (
                seg.loc[seg_dem.index, "x_atc"].values - seg["x_atc"].values[0]
            )
            ax_seg.plot(
                seg_dem_dist,
                seg_dem.values,
                color="gray",
                linewidth=1,
                label="DEM",
            )

        if show_aligned:
            seg_ald = seg["aligned_dem_height"].dropna()
            if not seg_ald.empty:
                seg_ald_dist = (
                    seg.loc[seg_ald.index, "x_atc"].values - seg["x_atc"].values[0]
                )
                ax_seg.plot(
                    seg_ald_dist,
                    seg_ald.values,
                    color="darkorange",
                    linewidth=1,
                    label="Aligned DEM",
                )

        # COP30 in segment
        cop30_col = "cop30.value"
        if cop30_col in seg.columns:
            seg_cop30 = seg[cop30_col].dropna()
            if not seg_cop30.empty:
                seg_cop30_dist = (
                    seg.loc[seg_cop30.index, "x_atc"].values - seg["x_atc"].values[0]
                )
                ax_seg.scatter(
                    seg_cop30_dist,
                    seg_cop30.values,
                    color="darkgoldenrod",
                    s=6,
                    alpha=0.6,
                    label="COP30",
                )

        if not seg_h.empty:
            seg_h_dist = seg.loc[seg_h.index, "x_atc"].values - seg["x_atc"].values[0]
            ax_seg.scatter(
                seg_h_dist,
                seg_h.values,
                color="steelblue",
                s=8,
                label="ICESat-2",
            )

        seg_dh = seg[dh_col].dropna()
        seg_med = np.nanmedian(seg_dh.values) if not seg_dh.empty else 0
        seg_nmad = _nmad(seg_dh.values) if len(seg_dh) >= 3 else 0
        title_parts = [f"{label_prefix} (Med={seg_med:+.1f} m, NMAD={seg_nmad:.1f} m)"]
        if show_aligned:
            seg_dh_ald = seg["icesat_minus_aligned_dem"].dropna()
            if not seg_dh_ald.empty:
                med_ald = np.nanmedian(seg_dh_ald.values)
                nmad_ald = _nmad(seg_dh_ald.values) if len(seg_dh_ald) >= 3 else 0
                title_parts.append(
                    f"Aligned (Med={med_ald:+.1f} m, NMAD={nmad_ald:.1f} m)"
                )
        ax_seg.set_title(
            "\n".join(title_parts),
            fontsize=9,
            color=color,
        )
        ax_seg.set_xlabel("Along-track distance (m)")
        ax_seg.set_ylabel("Elevation (m HAE)")
        ax_seg.grid(True, color="lightgray", linewidth=0.5, alpha=0.7)
        ax_seg.set_axisbelow(True)
        ax_seg.set_facecolor((*plt.matplotlib.colors.to_rgb(color), 0.05))

    def plot_best_worst_segments(
        self,
        resolved,
        seg_info,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot 1 km segments with better and worse agreement as a 1×2 figure.

        Creates a single-row, 2-column figure:
        - Column 1: Better agreement segment (lowest score)
        - Column 2: Worse agreement segment (highest score)

        Segment score is ``3·|median(dh)| + NMAD(dh)`` (see
        ``Icesat2Source._find_best_worst_segments``). Segment selection is
        based on the unaligned dh so best/worst segments remain comparable
        across the pre- and post-alignment variants of this plot.

        Parameters
        ----------
        resolved : tuple
            The ``(track, rgt, cycle, spot, track_count, track_date, dist,
            dh_vals)`` tuple from ``Icesat2Source._resolve_best_track``.
        seg_info : dict
            Best/worst segment extents from
            ``Icesat2Source._find_best_worst_segments``.
        plot_aligned : bool, optional
            Whether to overlay the aligned DEM heights and include aligned
            Median/NMAD in each segment title. Requires
            ``self.alt.aligned_dem_fn`` and the ``aligned_dem_height`` /
            ``icesat_minus_aligned_dem`` columns. Default False.
        save_dir : str or None, optional
            Directory to save figure, default is None
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        """
        track, rgt, cycle, spot, track_count, track_date, _, _ = resolved

        dh_col = "icesat_minus_dem"
        seg_best_color = "tab:blue"
        seg_worst_color = "tab:red"

        show_aligned = False
        if plot_aligned:
            if not self.alt.aligned_dem_fn:
                logger.warning(
                    "\nplot_aligned=True but no aligned DEM is available; "
                    "showing unaligned segments only.\n"
                )
            elif (
                "aligned_dem_height" not in track.columns
                or "icesat_minus_aligned_dem" not in track.columns
            ):
                logger.warning(
                    "\nAligned DEM columns missing from track; call "
                    "atl06sr_to_dem_dh() after setting aligned_dem_fn.\n"
                )
            else:
                show_aligned = True

        # --- 1×2 layout: better agreement | worse agreement ---
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            dpi=220,
            gridspec_kw={"wspace": 0.25},
        )
        ax_best, ax_worst = axes

        for ax_seg, mask, color, label_prefix in [
            (ax_best, seg_info["seg_best_mask"], seg_best_color, "Better agreement"),
            (ax_worst, seg_info["seg_worst_mask"], seg_worst_color, "Worse agreement"),
        ]:
            self._draw_segment_panel(
                ax_seg, track.loc[mask], color, label_prefix, show_aligned, dh_col
            )

        ax_best.legend(fontsize=7, loc="best")

        # Title
        title_str = f"RGT {rgt}, Cycle {cycle}, Spot {spot} ({track_date})"
        if track_count:
            title_str += f" — n={track_count}"
        if self._time_range_label:
            title_str += f"\n{self._time_range_label}"
        fig.suptitle(title_str, size=10)
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.97, wspace=0.25)
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
