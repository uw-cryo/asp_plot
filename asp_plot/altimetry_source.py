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
    def _std_outlier_mask(dh, n_sigma):
        """Boolean mask keeping dh values within ``n_sigma`` × std of the mean.

        Rows whose ``dh`` is NaN are kept (they carry no difference yet and
        must not be dropped). Returns ``None`` when the spread is degenerate
        (zero or non-finite std), signalling that no filtering should occur.

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
        mean_val = np.mean(valid)
        std_val = np.std(valid)
        if std_val == 0 or np.isnan(std_val):
            return None
        mask = (dh - mean_val).abs() <= n_sigma * std_val
        return mask | dh.isna()

    def _write_csv_to_directory(self, df, filename):
        """Write ``df`` to ``filename`` under the coordinator's directory.

        Returns the path written. Used by the pc_align CSV exporters, whose
        only difference is the columns they select.
        """
        csv_fn = os.path.join(self.alt.directory, filename)
        df.to_csv(csv_fn, header=True, index=False)
        return csv_fn
