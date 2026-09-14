"""Shared base for the altimetry sources.

Every altimetry source — ICESat-2 on Earth (:class:`Icesat2Source`) and the
planetary instruments LOLA/MOLA (:class:`LolaSource` / :class:`MolaSource`) —
samples an ASP DEM at a set of point locations, differences the altimetry
heights against it, removes coarse outliers, and exports a pc_align-ready CSV.
:class:`AltimetrySource` collects that shared machinery so each concrete source
only carries what is genuinely body-specific (the request/loader and the
height/datum conventions).

Sources hold a back reference to the coordinating
:class:`asp_plot.altimetry.Altimetry` instance and read the cross-cutting
``dem_fn`` / ``directory`` / ``aligned_dem_fn`` from it, so a single source of
truth describes the DEM under analysis.
"""

import os

import numpy as np
import rioxarray
import xarray as xr

# A residual this many NMADs from the median is not a DEM error but a blunder in
# the altimetry (cloud returns, a bad fit); dropped before the n_sigma × std cut
# so that a large cluster of them cannot inflate the std and defeat it.
GROSS_OUTLIER_NMAD = 30


class AltimetrySource:
    """Base for ICESat-2 and planetary altimetry sources.

    Parameters
    ----------
    alt : Altimetry
        The coordinating :class:`asp_plot.altimetry.Altimetry` instance.
    """

    def __init__(self, alt):
        self.alt = alt

    @staticmethod
    def _interp_dem_at_points(dem, points):
        """Bilinear-interpolate an open DEM at GeoDataFrame point locations.

        Parameters
        ----------
        dem : xarray.DataArray
            A DEM already opened (and squeezed) with rioxarray. Opening once
            and interpolating many times keeps the per-key ICESat-2 loop and
            the raw/aligned planetary passes from re-reading the raster.
        points : geopandas.GeoDataFrame
            Points to sample. Reprojected to the DEM CRS internally.

        Returns
        -------
        tuple of (numpy.ndarray, geopandas.GeoDataFrame)
            The sampled DEM values and the points reprojected to the DEM CRS.
            Callers that need the reprojected geometry (ICESat-2 stores it so
            downstream track geometry is in the DEM/working CRS) use the second
            element; the planetary path samples a throwaway copy and ignores it.
        """
        pts = points.to_crs(dem.rio.crs)
        x = xr.DataArray(pts.geometry.x.values, dims="z")
        y = xr.DataArray(pts.geometry.y.values, dims="z")
        sampled = dem.interp(x=x, y=y).values
        return sampled, pts

    @staticmethod
    def _open_dem(dem_fn):
        """Open a DEM as a squeezed, masked :class:`xarray.DataArray`."""
        return rioxarray.open_rasterio(dem_fn, masked=True).squeeze()

    @staticmethod
    def _outlier_mask(dh, n_sigma):
        """Boolean mask keeping dh values within ``n_sigma`` × std of the mean,
        after dropping gross outliers.

        Two cuts. First, anything farther than ``GROSS_OUTLIER_NMAD`` (30)
        normalized median absolute deviations from the median is dropped: no
        DEM error puts a point 30 NMADs out, but a cloud return does — one
        ICESat-2 pass of marine-layer cloud 150–200 m above the ground was a
        fifth of the sample over a coastal site, and a plain mean/std cut,
        its std inflated to ~70 m by that cluster, removed nothing and left
        every DEM with an RMSE near 85 m. Then the usual ``n_sigma`` × std cut
        about the mean of what survives, which is the cut every report has
        always applied and, absent gross contamination, gives the same result
        as before (the gross cut touches at most a few dozen points of the
        ~7000 over Atlanta).

        Rows whose ``dh`` is NaN are kept (they carry no difference yet and
        must not be dropped). Returns ``None`` — signalling that no filtering
        should occur — when there are no finite values or the spread is
        degenerate (zero or non-finite scale).

        Parameters
        ----------
        dh : pandas.Series
            Height differences (may contain NaN).
        n_sigma : float
            Number of standard deviations to allow about the mean.
        """
        valid = dh.dropna().values
        if valid.size == 0:
            return None
        median_val = np.median(valid)
        nmad_val = 1.4826 * np.median(np.abs(valid - median_val))
        if nmad_val == 0 or np.isnan(nmad_val):
            return None
        gross = (dh - median_val).abs() <= GROSS_OUTLIER_NMAD * nmad_val
        core = dh[gross].dropna().values
        mean_val = np.mean(core)
        std_val = np.std(core)
        if std_val == 0 or np.isnan(std_val):
            return None
        mask = gross & ((dh - mean_val).abs() <= n_sigma * std_val)
        return mask | dh.isna()

    def _write_csv_to_directory(self, df, filename):
        """Write ``df`` to ``filename`` under the coordinator's directory.

        Returns the path written. Used by the pc_align CSV exporters, whose
        only difference is the columns they select.
        """
        csv_fn = os.path.join(self.alt.directory, filename)
        df.to_csv(csv_fn, header=True, index=False)
        return csv_fn
