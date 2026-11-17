"""
Terrain attributes of a DEM at point locations.

:func:`slope_at_points` gives the slope of a DEM at altimetry points, which
:class:`asp_plot.dem_benchmark.DEMBenchmark` and
:class:`asp_plot.altimetry.Altimetry` use to break altimetry-minus-DEM
residuals down by slope class.
"""

import os

import numpy as np
import rasterio as rio
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from rasterio.windows import Window

#: Largest bounding box of points, in DEM pixels, read in one piece.
MAX_WINDOW_PIXELS = 16_000_000

#: Default slope classes in degrees (the classes of the #207 full-scene study).
SLOPE_BINS = (0, 5, 10, 20, 30, 90)

#: Default length scale in meters for slope under ICESat-2 points: about one
#: ATL06-SR segment (40 m long, 20 m apart), and the Copernicus DEM posting.
SLOPE_SCALE_M = 30


#: Column holding the slope in degrees sampled at altimetry points.
SLOPE_COLUMN = "slope_deg"


def slope_bin_edges(bins):
    """
    Slope class edges in degrees as a tuple of floats.

    Raises ``ValueError`` unless there are at least two increasing edges
    between 0 and 90.
    """
    edges = tuple(float(b) for b in bins)
    if len(edges) < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0 or edges[-1] > 90:
        raise ValueError(
            "slope_bins must be at least two increasing edges between 0 and "
            f"90 degrees; got {bins}."
        )
    return edges


def slope_bin_labels(edges):
    """Labels of the slope classes between ``edges``, such as ``"5–10°"``."""
    return [f"{lo:g}–{hi:g}°" for lo, hi in zip(edges[:-1], edges[1:])]


def _pixel_size_m(crs, transform, y):
    """
    Pixel width and height in meters, as scalars for a projected CRS or as
    arrays at latitudes ``y`` for a geographic one (from the CRS's ellipsoid,
    so a Moon or Mars DEM in degrees works too).
    """
    dx, dy = abs(transform.a), abs(transform.e)
    if crs is None or not crs.is_geographic:
        factor = 1.0
        if crs is not None:
            try:
                factor = float(crs.linear_units_factor[1])
            except Exception:  # unknown units: take them as meters
                factor = 1.0
        return dx * factor, dy * factor
    ellipsoid = CRS.from_user_input(crs.to_wkt()).ellipsoid
    a = ellipsoid.semi_major_metre
    inv_f = ellipsoid.inverse_flattening
    f = 1.0 / inv_f if inv_f else 0.0
    e2 = f * (2.0 - f)
    lat = np.radians(np.asarray(y, dtype=float))
    w = np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    meridional = a * (1.0 - e2) / w**3
    prime_vertical = a / w
    return (
        np.radians(dx) * prime_vertical * np.cos(lat),
        np.radians(dy) * meridional,
    )


def slope_at_points(dem_fn, points, scale=None):
    """
    Slope of a DEM in degrees at each point, by Horn's method.

    Horn's method is what ``gdaldem slope`` uses: the east and north
    gradients come from the 3 × 3 pixels around the pixel holding the point,
    with the centre row and column weighted twice. The DEM is not
    interpolated, so without ``scale`` the slope is that of the DEM at its
    own resolution: a 30 m Copernicus DEM gives the slope of the terrain
    over about one ATL06-SR segment, a 1 m DSM the slope of roofs and walls.

    Parameters
    ----------
    dem_fn : str
        DEM to take slope from. Heights are taken to be in meters; pixel
        sizes in degrees are converted to meters at each point's latitude.
    points : geopandas.GeoDataFrame
        Points, in any CRS (reprojected to the DEM's).
    scale : float, optional
        Length scale of the slope in meters. A DEM whose pixels are at most
        half this size is first averaged over blocks of whole pixels, the
        most that fit in ``scale``, so a 1 m DSM with ``scale=30`` gives
        the slope of its 30 m block means. A coarser DEM is used as it is.
        Default None: the DEM's own resolution.

    Returns
    -------
    numpy.ndarray
        Slope in degrees, NaN for a point outside the DEM or whose 3 × 3
        neighbourhood touches nodata or the DEM edge (as ``gdaldem`` without
        ``-compute_edges``).
    """
    n = len(points)
    slope = np.full(n, np.nan)
    if n == 0:
        return slope
    with rio.open(dem_fn) as ds:
        pts = points.to_crs(ds.crs) if ds.crs is not None else points
        xs = pts.geometry.x.to_numpy(dtype=float)
        ys = pts.geometry.y.to_numpy(dtype=float)
        transform, shape, z = ds.transform, (ds.height, ds.width), None
        factor = _block_factor(ds, scale, np.nanmedian(ys))
        if factor >= 2:
            averaged = _block_means(ds, xs, ys, factor)
            if averaged is None:
                return slope
            z, transform = averaged
            shape = z.shape
        rows, cols = (np.asarray(v, dtype=int) for v in rowcol(transform, xs, ys))
        inside = (
            (rows >= 1) & (rows < shape[0] - 1) & (cols >= 1) & (cols < shape[1] - 1)
        )
        if not inside.any():
            return slope
        r, c = rows[inside], cols[inside]
        r0, c0 = r.min() - 1, c.min() - 1
        height, width = r.max() - r0 + 2, c.max() - c0 + 2
        if z is not None:
            r0 = c0 = 0
        elif float(height) * float(width) <= MAX_WINDOW_PIXELS:
            # One read of the points' bounding box, then index the neighbours.
            z = _filled(ds.read(1, window=Window(c0, r0, width, height), masked=True))
        if z is not None:
            neighbours = np.stack(
                [z[r - r0 + dr, c - c0 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            )
        else:
            # Points spread over a large DEM: read each 3 × 3 window instead,
            # rather than a downsampled box that would change the slope.
            neighbours = np.stack(
                [
                    _filled(
                        ds.read(1, window=Window(ci - 1, ri - 1, 3, 3), masked=True)
                    ).ravel()
                    for ri, ci in zip(r, c)
                ],
                axis=1,
            )
        dx, dy = _pixel_size_m(ds.crs, transform, ys[inside])
    a, b, c_, d, e, f, g, h, i = neighbours
    dzdx = ((c_ + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c_)) / (8.0 * dy)
    values = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    values[~np.isfinite(e)] = np.nan
    slope[inside] = values
    return slope


def _block_factor(ds, scale, y):
    """
    Whole pixels of ``ds`` per side of the block averaged for ``scale``
    meters, at latitude or northing ``y``. 1: the DEM is used as it is.
    """
    if not scale:
        return 1
    size = _pixel_size_m(ds.crs, ds.transform, y)
    # 1e-6: 30 m over 1.2 m pixels is 25 pixels, not 24.999...
    return max(1, int(scale / max(np.max(size[0]), np.max(size[1])) + 1e-6))


def slope_pixel_size(dem_fn, scale=None):
    """
    The pixel size :func:`slope_at_points` takes slope at, in meters.

    Returns ``(size, averaged)``: the larger of the pixel's width and height
    (at the DEM's centre, for a DEM in degrees) after any averaging for
    ``scale``, and whether the DEM is averaged.
    """
    with rio.open(dem_fn) as ds:
        y = (ds.bounds.bottom + ds.bounds.top) / 2.0
        factor = _block_factor(ds, scale, y)
        size = _pixel_size_m(ds.crs, ds.transform, y)
    return float(max(np.max(size[0]), np.max(size[1])) * factor), factor >= 2


def slope_source_label(dem_fn, scale=None, name=None):
    """
    Axis label naming where slope came from, such as
    ``"Slope from cop30.tif (30 m)"`` or, for a DEM averaged for ``scale``,
    ``"Slope from run-DEM.tif (averaged to 30 m)"``. ``name`` replaces the
    file name.
    """
    size, averaged = slope_pixel_size(dem_fn, scale)
    name = name or os.path.basename(dem_fn)
    size = f"{size:.0f}" if size >= 100 else f"{size:.3g}"
    return f"Slope from {name} ({'averaged to ' if averaged else ''}{size} m)"


def _block_means(ds, xs, ys, factor):
    """
    The DEM around the points averaged over ``factor`` × ``factor`` pixel blocks.

    Reads the points' bounding box plus two blocks on every side, clipped to
    the raster, with GDAL's average resampling (nodata pixels are left out
    of a block's mean). GDAL averages while it reads, so only the
    block means are held here, but without overviews it reads every pixel of
    the box: about 15 s and 400 MB for a 130-megapixel DEM. Returns the array, NaN where a block has no valid
    pixel, and its affine transform, or None when no point is on the raster.
    """
    rows, cols = (np.asarray(v, dtype=int) for v in rowcol(ds.transform, xs, ys))
    on = (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
    if not on.any():
        return None
    pad = 2 * factor
    # Blocks start at multiples of ``factor`` from the raster's corner, so a
    # point's slope does not depend on which other points are passed.
    r0 = max(rows[on].min() - pad, 0) // factor * factor
    c0 = max(cols[on].min() - pad, 0) // factor * factor
    # Whole blocks only: a partial block at the raster's far edge is left out
    r1 = min(-(-(rows[on].max() + pad + 1) // factor), ds.height // factor) * factor
    c1 = min(-(-(cols[on].max() + pad + 1) // factor), ds.width // factor) * factor
    if r1 <= r0 or c1 <= c0:
        return None
    window = Window(c0, r0, c1 - c0, r1 - r0)
    out_shape = (window.height // factor, window.width // factor)
    z = _filled(
        ds.read(
            1,
            window=window,
            out_shape=out_shape,
            resampling=Resampling.average,
            masked=True,
        )
    )
    transform = ds.window_transform(window) * Affine.scale(factor, factor)
    return z, transform


def _filled(a):
    """A masked read as a float array with NaN where masked or non-finite."""
    return np.ma.masked_invalid(a.astype(float)).filled(np.nan)
