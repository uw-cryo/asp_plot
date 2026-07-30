"""Derived stereo geometry for RPC-only products (Cartosat-1, Deimos, ...).

ASP processes a long tail of products with ``-t rpc``: the camera model is a
set of rational polynomial coefficients embedded in the image header (NITF,
GeoTIFF) or delivered in a ``*_RPC.TXT`` sidecar, and nothing else. There is no
ephemeris, no attitude, no acquisition summary — nothing a reader could
*parse*. But as with ASTER (:mod:`asp_plot.sensors.aster`), "nothing to parse"
is not "nothing to plot": an RPC is a camera model, so the geometry can be
**derived** from it (#177).

The derivation rests on one observation: projecting the *same pixel* to the
ground at two different heights traces that pixel's look ray. From there,

- the **footprint** is the image border projected at ``HEIGHT_OFF``;
- the **satellite azimuth/elevation** is the centre pixel's look ray expressed
  in the local east/north/up frame at its ground point — the DigitalGlobe
  ``MEANSATAZ``/``MEANSATEL`` convention the pair math (convergence angle, B:H,
  BIE) already speaks;
- the **perspective centre** — where the satellite actually was — is the
  intersection of two look rays from opposite ends of the same image line, and
  gives the off-nadir view angle and an approximate position track;
- the **GSD** is the ground spacing of one pixel at the scene centre.

Accuracy is not assumed, it is measured. Every derived quantity is pinned in
``tests/test_sensors.py`` against the vendor's own numbers in the committed
WorldView camera XMLs, whose ``RPB`` blocks carry real RPC00B coefficients for
scenes whose ``MEANSATAZ``/``MEANSATEL``/``MEANOFFNADIRVIEWANGLE``/
``MEANPRODUCTGSD`` are recorded independently: azimuth matches to 0.01°,
elevation and off-nadir to 0.15°, GSD to 1 cm.

Two things are deliberately **not** derived:

- ``meanintrackviewangle`` / ``meancrosstrackviewangle``. Splitting the
  off-nadir angle into along- and across-track parts needs a velocity
  direction, and the only one available here is the drift of the recovered
  perspective centres — over the ~15 km of track one scene spans, their
  kilometre-level noise tilts that direction by 8–10° (measured against the
  vendor ephemeris). The total off-nadir angle needs no velocity and is
  accurate, so it is reported and the split is left as "not provided" (#163).
- Time. RPCs carry no timestamps. ``date`` is recovered from the image
  header when the container records one (NITF ``IDATIM``, TIFF
  ``DateTime``) and is otherwise None, which the pair code renders as "N/A".

``att_df`` is None (no attitude exists) and ``eph_gdf``, when the perspective
centres are recoverable, is indexed by image **line** rather than time — the
same contract ASTER established.
"""

import logging
import os
import re
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.rpc import RPC
from rasterio.transform import RPCTransformer
from shapely import Polygon

from asp_plot.sensors.base import (
    IMAGE_EXTENSIONS,
    SensorMetadata,
    _ecef_height,
    _ecef_to_lonlat,
    _enu_basis,
    _lonlat_to_ecef,
    fill_scene_defaults,
    list_candidate_images,
    resolve_input_files,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Number of samples per image edge when tracing the footprint. An RPC is a
# polynomial fit, so the projected image border is gently curved rather than a
# quadrilateral; a handful of points per edge captures that at no real cost.
_FOOTPRINT_EDGE_SAMPLES = 11

# Number of image lines at which the perspective centre is recovered for the
# position track.
_EPHEMERIS_SAMPLES = 11

# Height separation (m) used to trace a pixel's look ray when the RPC's own
# HEIGHT_SCALE is degenerate. A ray is defined by two points, and they must be
# far enough apart that the RPC's fit residual does not dominate the direction.
_DEFAULT_HEIGHT_SPREAD = 500.0

# Two look rays from the same image line must converge on the satellite. The
# recovered point is rejected when they pass this far apart, relative to the
# range: rays that are effectively parallel (an already-orthorectified raster
# that kept its RPCs, say) intersect nowhere meaningful, and a bogus position
# would silently corrupt the off-nadir angle. Real products come in three
# orders of magnitude under this (measured: 75 m at a 780 km range).
_MAX_RAY_MISS_FRACTION = 0.01

# Sidecar RPC files: GDAL itself picks up ``<stem>_RPC.TXT`` when opening an
# image, but Cartosat-1 deliveries name theirs ``<stem>_RPC_ORG.TXT`` — ASP
# works around this by renaming the file (``StereoSession.cc``). Reading the
# sidecar directly means neither we nor the user has to.
_SIDECAR_RE = re.compile(r"_rpc(_org)?\.txt$", re.IGNORECASE)

# RPC00B field names, as written in a sidecar TXT, mapped to the rasterio RPC
# constructor's scalar arguments.
_SIDECAR_SCALARS = {
    "LINE_OFF": "line_off",
    "SAMP_OFF": "samp_off",
    "LAT_OFF": "lat_off",
    "LONG_OFF": "long_off",
    "HEIGHT_OFF": "height_off",
    "LINE_SCALE": "line_scale",
    "SAMP_SCALE": "samp_scale",
    "LAT_SCALE": "lat_scale",
    "LONG_SCALE": "long_scale",
    "HEIGHT_SCALE": "height_scale",
}
_SIDECAR_COEFFS = {
    "LINE_NUM_COEFF": "line_num_coeff",
    "LINE_DEN_COEFF": "line_den_coeff",
    "SAMP_NUM_COEFF": "samp_num_coeff",
    "SAMP_DEN_COEFF": "samp_den_coeff",
}
# "KEY: value units", e.g. "LINE_OFF: +013824.00 pixels".
_SIDECAR_LINE_RE = re.compile(r"^\s*([A-Za-z_0-9]+)\s*:\s*([-+0-9.eE]+)")

_warned_sidecars = set()


def _parse_rpc_sidecar(path):
    """Parse an RPC00B ``*_RPC.TXT`` sidecar into a :class:`rasterio.rpc.RPC`.

    Returns None if the file is not a complete RPC00B block, so a stray text
    file next to an image is simply not claimed.
    """
    scalars, coeffs = {}, {}
    try:
        with open(path, errors="ignore") as f:
            for line in f:
                match = _SIDECAR_LINE_RE.match(line)
                if not match:
                    continue
                key, value = match.group(1).upper(), float(match.group(2))
                if key in _SIDECAR_SCALARS:
                    scalars[_SIDECAR_SCALARS[key]] = value
                    continue
                # Coefficients are written one per line, 1-indexed:
                # "LINE_NUM_COEFF_7: +1.234e-05".
                stem, _, index = key.rpartition("_")
                if stem in _SIDECAR_COEFFS and index.isdigit():
                    coeffs.setdefault(_SIDECAR_COEFFS[stem], {})[int(index)] = value
    except (OSError, ValueError):
        return None

    if set(scalars) != set(_SIDECAR_SCALARS.values()):
        return None
    for name in _SIDECAR_COEFFS.values():
        indices = coeffs.get(name, {})
        if set(indices) != set(range(1, 21)):
            return None
        scalars[name] = [indices[i] for i in range(1, 21)]
    return RPC(**scalars)


def _sidecar_for(image_fn):
    """Find an RPC sidecar belonging to ``image_fn``, or None.

    Matches ``<stem>_RPC.TXT`` and Cartosat-1's ``<stem>_RPC_ORG.TXT``,
    case-insensitively, in the image's own directory.
    """
    directory = os.path.dirname(os.path.abspath(image_fn))
    stem = os.path.splitext(os.path.basename(image_fn))[0].lower()
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return None
    for name in names:
        match = _SIDECAR_RE.search(name)
        if match and name[: match.start()].lower() == stem:
            return os.path.join(directory, name)
    return None


def read_rpc(image_fn):
    """Read the RPC camera model and grid size of an image.

    Tries the image header first (GDAL exposes embedded RPCs, and finds a
    ``<stem>_RPC.TXT`` sidecar on its own), then falls back to parsing a
    sidecar directly — which is what turns Cartosat-1's ``_RPC_ORG.TXT``
    deliveries into readable products.

    Rasters that are already map-projected are rejected even when they carry
    RPCs: an RPC describes the *raw* image grid, so on an orthorectified
    product it no longer corresponds to the pixels it would be evaluated
    against.

    Parameters
    ----------
    image_fn : str
        Path to an image file.

    Returns
    -------
    tuple or None
        ``(rpc, width, height, tags)`` — a :class:`rasterio.rpc.RPC`, the
        image's pixel dimensions and its metadata tags — or None if the file
        is not an unprojected raster carrying RPCs.
    """
    if os.path.splitext(image_fn)[1].lower() not in IMAGE_EXTENSIONS:
        return None
    try:
        # An image whose camera model lives in a sidecar has no georeferencing
        # of any kind until that sidecar is read, which is the normal state of
        # affairs here rather than something to warn a user about.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(image_fn) as src:
                if src.crs is not None:
                    return None
                rpc, width, height = src.rpcs, src.width, src.height
                tags = dict(src.tags())
    except Exception:
        # Unreadable, or not a raster at all: not this reader's file.
        return None

    if not rpc:
        sidecar = _sidecar_for(image_fn)
        if sidecar is None:
            return None
        rpc = _parse_rpc_sidecar(sidecar)
        if rpc is None:
            return None
        if _SIDECAR_RE.search(sidecar).group(1) and sidecar not in _warned_sidecars:
            _warned_sidecars.add(sidecar)
            logger.warning(
                "Read RPCs from the non-standard sidecar %s. GDAL only picks "
                "up '<image>_RPC.TXT' automatically, so this path is exercised "
                "by Cartosat-1 deliveries and has not been validated against a "
                "real one -- please report how it went at "
                "https://github.com/uw-cryo/asp_plot/issues/new",
                os.path.basename(sidecar),
            )
    return rpc, width, height, tags


def _acquisition_date(tags):
    """Recover an acquisition time from image header tags, or None.

    RPCs carry no time, but the containers that hold them sometimes do: NITF
    records ``IDATIM`` (``CCYYMMDDhhmmss``) and TIFF ``DateTime``
    (``YYYY:MM:DD HH:MM:SS``). Both are best-effort — a product with neither
    is simply dateless (#163).
    """
    idatim = tags.get("NITF_IDATIM", "").strip()
    if len(idatim) >= 14 and idatim[:14].isdigit():
        try:
            return datetime.strptime(idatim[:14], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    tifftag = tags.get("TIFFTAG_DATETIME", "").strip()
    try:
        return datetime.strptime(tifftag, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _closest_approach(p1, d1, p2, d2):
    """Midpoint of the common perpendicular between two rays, and their miss.

    Parameters
    ----------
    p1, p2 : numpy.ndarray
        Ray origins (ECEF, metres).
    d1, d2 : numpy.ndarray
        Unit ray directions (ECEF).

    Returns
    -------
    tuple of (numpy.ndarray, float) or None
        The midpoint of the two rays' closest approach and the distance
        between them there, or None when the rays are parallel.
    """
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d_, e_ = d1 @ w0, d2 @ w0
    denom = a * c - b * b
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return None
    q1 = p1 + ((b * e_ - c * d_) / denom) * d1
    q2 = p2 + ((a * e_ - b * d_) / denom) * d2
    return (q1 + q2) / 2.0, float(np.linalg.norm(q1 - q2))


class _RpcGeometry:
    """The stereo geometry derived from one image's RPC camera model.

    Answers the geometric questions the scene dict needs: where the scene is
    (:meth:`footprint`), where the satellite was relative to it
    (:meth:`view_angles`, :meth:`ephemeris`), and how big a pixel is
    (:meth:`gsd`).

    Attributes
    ----------
    rpc : rasterio.rpc.RPC
        The image's rational polynomial camera model.
    width, height : int
        Image size in pixels (samples, lines).
    """

    def __init__(self, rpc, width, height):
        self.rpc = rpc
        self.width = int(width)
        self.height = int(height)
        if self.width < 2 or self.height < 2:
            raise ValueError(
                f"Image is {self.width}x{self.height} pixels; RPC geometry "
                "needs a real image grid to trace."
            )
        spread = float(rpc.height_scale)
        self._height_spread = spread if spread > 0 else _DEFAULT_HEIGHT_SPREAD

    @property
    def center_pixel(self):
        """The image centre as a (line, sample) pair."""
        return self.height / 2.0, self.width / 2.0

    def _project(self, rows, cols, heights):
        """Project pixels to (lon, lat) at the given heights.

        ``offset="ul"`` so a pixel coordinate means the same thing here as in
        the image border traced by :meth:`footprint` — rasterio otherwise
        shifts by half a pixel to the pixel centre.
        """
        with RPCTransformer(self.rpc) as transformer:
            xs, ys = transformer.xy(
                list(rows), list(cols), zs=list(heights), offset="ul"
            )
        lon, lat = np.atleast_1d(xs).astype(float), np.atleast_1d(ys).astype(float)
        if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
            raise ValueError(
                "The RPC camera model does not project this image's pixels to "
                "valid ground coordinates; its coefficients are unusable."
            )
        return lon, lat

    def look_ray(self, row, col):
        """Trace one pixel's look ray.

        Projecting the same pixel at two heights gives two points on its ray;
        the higher one is nearer the satellite, so their difference points up
        the ray.

        Returns
        -------
        tuple of numpy.ndarray
            The pixel's ground point at ``HEIGHT_OFF`` (ECEF, metres) and the
            unit vector from it toward the satellite.
        """
        h_low = self.rpc.height_off - self._height_spread / 2.0
        h_high = self.rpc.height_off + self._height_spread / 2.0
        heights = [h_low, self.rpc.height_off, h_high]
        lon, lat = self._project([row] * 3, [col] * 3, heights)
        points = _lonlat_to_ecef(lon, lat, heights)
        direction = points[2] - points[0]
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError(
                "The RPC camera model projects this pixel to the same ground "
                "point at every height, so it defines no look direction."
            )
        return points[1], direction / norm

    def perspective_center(self, row):
        """Recover the satellite position for one image line, or None.

        Two look rays from opposite ends of a line meet at the perspective
        centre. The intersection is only meaningful if they really do converge,
        so a result whose rays miss each other by more than
        :data:`_MAX_RAY_MISS_FRACTION` of the range — or which lands below the
        ellipsoid — is rejected rather than reported.
        """
        try:
            p1, d1 = self.look_ray(row, 0)
            p2, d2 = self.look_ray(row, self.width)
        except ValueError:
            return None
        result = _closest_approach(p1, d1, p2, d2)
        if result is None:
            return None
        satellite, miss = result
        if miss > _MAX_RAY_MISS_FRACTION * np.linalg.norm(satellite - p1):
            return None
        if not np.isfinite(satellite).all() or _ecef_height(satellite) <= 0:
            return None
        return satellite

    def footprint(self):
        """Scene footprint in EPSG:4326, traced around the image border."""
        line = np.linspace(0, self.height, _FOOTPRINT_EDGE_SAMPLES)
        sample = np.linspace(0, self.width, _FOOTPRINT_EDGE_SAMPLES)
        rows = np.concatenate(
            [np.zeros_like(sample), line, np.full_like(sample, self.height), line[::-1]]
        )
        cols = np.concatenate(
            [sample, np.full_like(line, self.width), sample[::-1], np.zeros_like(line)]
        )
        lon, lat = self._project(rows, cols, np.full(rows.size, self.rpc.height_off))
        polygon = Polygon(zip(lon, lat))
        if not polygon.is_valid:
            raise ValueError(
                "The image border projected through the RPC is not a simple "
                "polygon, so this camera model does not describe a normal "
                "scene footprint. Please report this product at "
                "https://github.com/uw-cryo/asp_plot/issues/new"
            )
        return polygon

    def view_angles(self):
        """Satellite view geometry at the scene centre.

        Azimuth and elevation are measured at the ground point in its local
        ENU frame — the DigitalGlobe ``MEANSATAZ``/``MEANSATEL`` convention the
        pair math expects. The off-nadir angle is measured at the recovered
        perspective centre against geodetic nadir, and is omitted when that
        cannot be recovered. The in-track/cross-track split of the off-nadir
        angle is never derived; see the module docstring.

        Returns
        -------
        dict
            ``meansataz`` and ``meansatel``, plus ``meanoffnadirviewangle``
            when the perspective centre is recoverable, in degrees.
        """
        row, col = self.center_pixel
        ground, to_sat = self.look_ray(row, col)

        lon, lat = _ecef_to_lonlat(ground)
        east, north, up = _enu_basis(float(lon), float(lat))
        angles = {
            "meansataz": float(
                np.round(
                    np.degrees(np.arctan2(to_sat @ east, to_sat @ north)) % 360.0, 2
                )
            ),
            "meansatel": float(
                np.round(np.degrees(np.arcsin(np.clip(to_sat @ up, -1.0, 1.0))), 2)
            ),
        }

        satellite = self.perspective_center(row)
        if satellite is not None:
            sat_lon, sat_lat = _ecef_to_lonlat(satellite)
            _, _, sat_up = _enu_basis(float(sat_lon), float(sat_lat))
            look = -to_sat
            angles["meanoffnadirviewangle"] = float(
                np.round(np.degrees(np.arccos(np.clip(look @ -sat_up, -1.0, 1.0))), 2)
            )
        return angles

    def gsd(self):
        """Mean ground sample distance (m) at the scene centre.

        Averages the ground spacing of one pixel in the sample and line
        directions, measured across the centre pixel.
        """
        row, col = self.center_pixel
        rows = [row, row, row - 0.5, row + 0.5]
        cols = [col - 0.5, col + 0.5, col, col]
        lon, lat = self._project(rows, cols, [self.rpc.height_off] * 4)
        points = _lonlat_to_ecef(lon, lat, np.full(4, self.rpc.height_off))
        across = np.linalg.norm(points[1] - points[0])
        along = np.linalg.norm(points[3] - points[2])
        return float(np.round((across + along) / 2.0, 2))

    def ephemeris(self):
        """Recovered satellite positions as a GeoDataFrame, or None.

        Indexed by image **line** rather than time (RPCs carry no timestamps),
        with NaN velocity and covariance columns so consumers test for
        "provided" exactly as they do for every other sensor. Returns None
        unless the perspective centre is recoverable at every sampled line —
        a partial track would misrepresent both the position panel and the
        pair's asymmetry angle.

        These are recovered positions, not an ephemeris: they carry
        kilometre-level error (see the module docstring), which is small
        against the ~700 km range that sets the viewing geometry but far too
        coarse to read as a trajectory.
        """
        lines = np.linspace(0, self.height, _EPHEMERIS_SAMPLES)
        positions = [self.perspective_center(line) for line in lines]
        if any(p is None for p in positions):
            return None

        eph_df = pd.DataFrame(np.array(positions), columns=["x", "y", "z"])
        for name in ("dx", "dy", "dz"):
            eph_df[name] = np.nan
        for name in ("11", "12", "13", "22", "23", "33"):
            eph_df[f"cov_{name}"] = np.nan
        eph_df["line"] = lines
        eph_df.set_index("line", inplace=True)
        return gpd.GeoDataFrame(
            eph_df,
            geometry=gpd.points_from_xy(eph_df["x"], eph_df["y"], eph_df["z"]),
            crs="EPSG:4978",
        )


class RpcMetadata(SensorMetadata):
    """Metadata reader for images whose only camera model is an RPC.

    Covers everything ASP runs with ``-t rpc`` — Cartosat-1, Deimos, and the
    tail of commercial products delivered with rational polynomials and
    nothing else — as one reader, because the camera model *is* the format.

    Unlike every other reader in this package, the file it claims is the
    image, not a sidecar XML. That makes it a ``fallback`` reader: a WorldView
    or Pléiades delivery ships images alongside its camera XMLs and those
    images carry RPCs too, so this reader is only consulted once every
    XML-based reader has declined the input at every search depth.

    See the module docstring for what is derived and what is unavailable.
    """

    name = "RPC-only"
    fallback = True

    def __init__(self, directory=None, image_list=None):
        """
        Initialize an RPC metadata reader.

        Parameters
        ----------
        directory : str, optional
            Path to a directory containing images carrying RPCs.
        image_list : list of str, optional
            Explicit list of images to use instead of discovering them from
            ``directory``.

        Raises
        ------
        ValueError
            If neither ``directory`` nor ``image_list`` is given, or if no
            image carrying an RPC camera model is found.
        """
        self.directory, self.image_list = resolve_input_files(
            type(self), directory, image_list
        )
        if not self.image_list:
            raise ValueError(
                "\n\nMissing images carrying RPC camera models. "
                "Cannot extract metadata without these.\n\n"
            )

    @classmethod
    def _is_camera_file(cls, path):
        """True if ``path`` is an unprojected raster carrying an RPC model.

        Costlier than the other readers' checks — it opens the raster header —
        so discovery filters by extension first, and this reader is consulted
        last.
        """
        return read_rpc(path) is not None

    @classmethod
    def _discover_camera_files(cls, directory, recursive=True):
        """Discover images carrying RPCs in ``directory``.

        Overrides the base implementation's ``*.xml`` search: this reader's
        camera files are images (:func:`asp_plot.sensors.base.list_candidate_images`).
        """
        return cls._filter_camera_files(
            list_candidate_images(directory, recursive=recursive)
        )

    def get_scene_dicts(self):
        """Return one sensor-agnostic scene dict per image."""
        return [self.get_scene_dict(image) for image in self.image_list]

    def get_scene_dict(self, image_fn, geteph=True):
        """
        Get a dictionary of metadata for one RPC image.

        Parameters
        ----------
        image_fn : str
            Path to the image carrying the RPC camera model.
        geteph : bool, optional
            Whether to include the trajectory block, default is True.

        Returns
        -------
        dict
            Sensor-agnostic scene dict (see the package docstring).

        Raises
        ------
        ValueError
            If ``image_fn`` carries no usable RPC camera model.
        """
        read = read_rpc(image_fn)
        if read is None:
            raise ValueError(
                f"\n\n{image_fn} is not an unprojected raster carrying an RPC "
                "camera model.\n\n"
            )
        rpc, width, height, tags = read
        geometry = _RpcGeometry(rpc, width, height)
        geom = geometry.footprint()

        d = {
            # The schema's "xml_fn" slot holds whatever file the scene's
            # geometry came from; here that is the image itself.
            "xml_fn": image_fn,
            # No catalog ID exists in RPC metadata, so scenes are identified by
            # image name -- which is also how ASP's own command lines name them.
            "catid": os.path.splitext(os.path.basename(image_fn))[0],
            "sensor": tags.get("NITF_ISORCE", "").strip() or "RPC",
            "date": _acquisition_date(tags),
            "geom": geom,
            "meanproductgsd": geometry.gsd(),
            **geometry.view_angles(),
        }

        if geteph:
            eph_gdf = geometry.ephemeris()
            # Omitted rather than None when unrecoverable: consumers test for
            # the key (the pair asymmetry angle) or annotate its absence.
            if eph_gdf is not None:
                d["eph_gdf"] = eph_gdf
            # An RPC records no attitude at all; None is the documented "not
            # provided" value (see the package docstring).
            d["att_df"] = None
            d["fp_gdf"] = gpd.GeoDataFrame(
                {"idx": [0], "geometry": [geom]},
                geometry="geometry",
                crs="EPSG:4326",
            )

        return fill_scene_defaults(d)
