"""Derived stereo geometry for ASTER ``gen_aster`` camera XMLs.

Every other reader in this package *parses* the geometry a vendor wrote down.
ASTER has none to parse: ASP's ``gen_aster`` writes an ``<isd>`` camera file
(``ASTER_XML.cc``) with no timestamps, no attitude, no view- or sun-angle
summary and no footprint corners. What it does carry is the raw material to
**derive** the geometry:

- ``SAT_POS`` — satellite ECEF positions, one per row of the lattice
- ``WORLD_SIGHT_VECTOR`` — the ECEF look direction at each lattice point
- ``LATTICE_POINT`` — the (sample, line) pixel grid those vectors attach to,
  in row-major order
- ``IMAGE_COLS`` / ``IMAGE_ROWS`` — the image extent the lattice spans (the
  lattice deliberately overshoots it by a few hundred lines)

Intersecting each look ray with the WGS84 ellipsoid turns that into a ground
lattice, and everything the stereo-geometry code needs follows from it: the
footprint (the image border traced through the ground lattice), the satellite
azimuth/elevation at the scene center, the off-nadir/in-track/cross-track view
angles at the spacecraft, and the ground sample distance.

Two derivations are worth stating plainly, because they set the accuracy of
everything downstream:

- Rays are intersected with the **ellipsoid** (h=0), not with terrain. The
  footprint therefore shifts by ``h * tan(off-nadir)`` — under a kilometre for
  the nadir band and a couple for the backward one over high terrain. View
  angles are unaffected: moving the ground point along its own ray does not
  change the ray's direction, only the (negligible) local vertical.
- Time is not derivable. There is no timestamp anywhere in the camera file, so
  ``eph_gdf`` is indexed by lattice **line number** rather than time, and
  ``att_df`` is None — ASTER reports no attitude at all (see the
  :mod:`asp_plot.sensors` package docstring for that contract). ``date`` is
  recovered from a neighbouring ``AST_L1A_*`` granule name when one is present,
  which means both bands of a pair report the same acquisition time even though
  the backward look trails the nadir look by roughly a minute.

Unlike the DIMAP-family readers, this one is validated against real data: the
committed ``gen_aster`` fixtures reproduce ASTER's published geometry — a
27.6° backward in-track pointing for band 3B, ~15 m GSD, and a footprint
matching the camera file's own RPC bounding box (#175).
"""

import logging
import os
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from shapely import Polygon

from asp_plot.sensors.base import (
    SensorMetadata,
    fill_scene_defaults,
    resolve_input_files,
)
from asp_plot.utils import aster_datetime_from_name

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# WGS84 ellipsoid, the surface the look rays are intersected with. ASP's own
# ASTER model triangulates against the same datum.
WGS84_A = 6378137.0
WGS84_B = WGS84_A * (1.0 - 1.0 / 298.257223563)

# The tags that identify a gen_aster camera file. The ``<isd>`` root is shared
# with DigitalGlobe-heritage XMLs (which carry IMD/EPH/ATT blocks instead), so
# the lattice tags are what tell the two apart in both directions.
_ASTER_TAGS = ("LATTICE_POINT", "SIGHT_VECTOR", "SAT_POS")

# Number of samples per image edge when tracing the footprint. The scene border
# of a pushbroom is gently curved, so a handful of points per edge captures it
# far better than four corners at no meaningful cost.
_FOOTPRINT_EDGE_SAMPLES = 11

_ECEF_TO_LONLAT = None


def _ecef_to_lonlat(xyz):
    """Convert ECEF coordinates to (lon, lat) degrees.

    Parameters
    ----------
    xyz : numpy.ndarray
        Array of shape ``(..., 3)`` of ECEF x, y, z in metres.

    Returns
    -------
    tuple of numpy.ndarray
        Longitude and latitude in degrees, each shaped like ``xyz[..., 0]``.
    """
    global _ECEF_TO_LONLAT
    if _ECEF_TO_LONLAT is None:
        _ECEF_TO_LONLAT = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    xyz = np.asarray(xyz, dtype=float)
    shape = xyz.shape[:-1]
    flat = xyz.reshape(-1, 3)
    lon, lat, _ = _ECEF_TO_LONLAT.transform(flat[:, 0], flat[:, 1], flat[:, 2])
    return np.asarray(lon).reshape(shape), np.asarray(lat).reshape(shape)


def _enu_basis(lon, lat):
    """Return the local east/north/up unit vectors at a geodetic position.

    Parameters
    ----------
    lon, lat : float
        Geodetic longitude and latitude in degrees.

    Returns
    -------
    tuple of numpy.ndarray
        The east, north and up unit vectors, in ECEF.
    """
    lon, lat = np.radians(lon), np.radians(lat)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    )
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return east, north, up


def _ray_ellipsoid_intersection(origins, directions):
    """Intersect look rays with the WGS84 ellipsoid.

    Solves ``|(P + t*D)|_ellipsoid = 1`` for the near root, i.e. the first
    surface point along each ray. Broadcasting: ``origins`` and ``directions``
    need only be broadcast-compatible, so a whole lattice row can share one
    satellite position.

    Parameters
    ----------
    origins : numpy.ndarray
        Ray origins (ECEF, metres), shape ``(..., 3)``.
    directions : numpy.ndarray
        Ray directions (ECEF), shape ``(..., 3)``. Need not be normalized.

    Returns
    -------
    numpy.ndarray
        Ellipsoid intersection points (ECEF, metres), broadcast shape.

    Raises
    ------
    ValueError
        If any ray misses the ellipsoid, or hits it only *behind* its origin
        (a look vector pointing away from the Earth). Both mean the look
        vectors and positions are not the consistent pair this reader assumes,
        and silently returning the far or backward root would put the scene
        somewhere plausible-looking but wrong.
    """
    scale = np.array([1.0 / WGS84_A, 1.0 / WGS84_A, 1.0 / WGS84_B])
    p = np.asarray(origins, dtype=float) * scale
    d = np.asarray(directions, dtype=float) * scale

    a = np.sum(d * d, axis=-1)
    b = 2.0 * np.sum(p * d, axis=-1)
    c = np.sum(p * p, axis=-1) - 1.0
    discriminant = b * b - 4.0 * a * c
    with np.errstate(invalid="ignore"):
        t = (-b - np.sqrt(discriminant)) / (2.0 * a)
    if np.any(discriminant < 0) or np.any(t < 0):
        raise ValueError(
            "Sight vectors do not intersect the Earth ellipsoid ahead of the "
            "satellite: the camera file's SAT_POS and WORLD_SIGHT_VECTOR "
            "blocks are inconsistent."
        )
    return np.asarray(origins, dtype=float) + t[..., np.newaxis] * np.asarray(
        directions, dtype=float
    )


def _float_rows(root, tag, width):
    """Parse a whitespace-separated numeric block into an ``(N, width)`` array.

    The lattice blocks are free text: one record per line, blank lines between
    lattice rows.

    Raises
    ------
    ValueError
        If the tag is missing, empty, or its rows are not ``width`` wide.
    """
    text = root.findtext(tag)
    if text is None:
        raise ValueError(f"gen_aster camera file has no <{tag}> block.")
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows or any(len(r) != width for r in rows):
        raise ValueError(
            f"<{tag}> is not a table of {width} numbers per line; got "
            f"{len(rows)} rows with widths {sorted({len(r) for r in rows})}."
        )
    try:
        return np.array(rows, dtype=float)
    except ValueError as exc:
        raise ValueError(f"<{tag}> holds non-numeric values.") from exc


def _is_aster_isd(path):
    """True if ``path`` is a gen_aster camera XML (``<isd>`` + lattice blocks).

    Cheap ``iterparse`` peek on start events only: tag presence is enough, so
    the (large) lattice text never has to be read, and the scan stops as soon
    as the signature is complete.
    """
    try:
        wanted = set(_ASTER_TAGS)
        seen = set()
        root_seen = False
        for _, el in ET.iterparse(path, events=("start",)):
            if not root_seen:
                if el.tag != "isd":
                    return False
                root_seen = True
                continue
            seen.add(el.tag)
            if wanted.issubset(seen):
                return True
        return False
    except (ET.ParseError, OSError):
        return False


def _acquisition_date(xml_fn):
    """Recover the acquisition date from a neighbouring ASTER L1A granule name.

    The camera file itself has no timestamp, but ``gen_aster`` is run on an
    ``AST_L1A_*`` granule whose name encodes the capture time and which
    normally still sits beside its output. The camera file's own name is
    checked first, then its directory.

    Returns None when no granule name is found — the scene is then dateless,
    which the pair code renders as "N/A" (#163).
    """
    path = os.path.abspath(xml_fn)
    directory = os.path.dirname(path)
    names = [os.path.basename(path)]
    try:
        names.extend(sorted(os.listdir(directory)))
    except OSError:
        pass
    for name in names:
        date = aster_datetime_from_name(name)
        if date is not None:
            return date
    return None


class _AsterLattice:
    """The ground geometry derived from one gen_aster camera file.

    Parses the lattice blocks once and intersects every look ray with the
    ellipsoid, then answers the geometric questions the scene dict needs:
    where the scene is (:meth:`footprint`), where the satellite was relative to
    it (:meth:`view_angles`), and how big a pixel is (:meth:`gsd`).

    Attributes
    ----------
    rows, cols : numpy.ndarray
        The lattice's unique line and sample coordinates (image pixels).
    sat_pos : numpy.ndarray
        ``(len(rows), 3)`` ECEF satellite positions, one per lattice row.
    ground : numpy.ndarray
        ``(len(rows), len(cols), 3)`` ECEF ellipsoid intersection points.
    image_cols, image_rows : int
        The image extent, which the lattice overshoots by design.
    """

    def __init__(self, root):
        lattice = _float_rows(root, "LATTICE_POINT", 2)
        sight = _float_rows(root, "WORLD_SIGHT_VECTOR", 3)
        self.sat_pos = _float_rows(root, "SAT_POS", 3)

        # LATTICE_POINT is "sample line", listed row-major (all samples of one
        # line, then the next line), and SAT_POS holds one position per line —
        # a pushbroom's position varies with line, not sample.
        self.cols = np.unique(lattice[:, 0])
        self.rows = np.unique(lattice[:, 1])
        n_rows, n_cols = len(self.rows), len(self.cols)

        if n_rows * n_cols != len(lattice):
            raise ValueError(
                f"LATTICE_POINT is not a complete grid: {len(lattice)} points "
                f"for {n_rows} lines x {n_cols} samples."
            )
        if len(sight) != len(lattice):
            raise ValueError(
                f"WORLD_SIGHT_VECTOR has {len(sight)} entries but "
                f"LATTICE_POINT has {len(lattice)}; they must correspond."
            )
        if len(self.sat_pos) != n_rows:
            raise ValueError(
                f"SAT_POS has {len(self.sat_pos)} positions but the lattice has "
                f"{n_rows} lines; expected one satellite position per line."
            )
        grid = lattice.reshape(n_rows, n_cols, 2)
        if not (
            np.array_equal(grid[:, :, 1], np.tile(self.rows[:, None], (1, n_cols)))
            and np.array_equal(grid[:, :, 0], np.tile(self.cols[None, :], (n_rows, 1)))
        ):
            raise ValueError(
                "LATTICE_POINT is not in row-major order over a regular "
                "(line x sample) grid, which the ground lattice assumes."
            )

        self.ground = _ray_ellipsoid_intersection(
            self.sat_pos[:, None, :], sight.reshape(n_rows, n_cols, 3)
        )
        self._interp = RegularGridInterpolator((self.rows, self.cols), self.ground)

        self.image_cols = int(float(root.findtext("IMAGE_COLS")))
        self.image_rows = int(float(root.findtext("IMAGE_ROWS")))

    def ground_at(self, row, col):
        """ECEF ground point at a (line, sample) pixel coordinate."""
        return self._interp(np.array([[row, col]]))[0]

    def sat_at(self, row):
        """ECEF satellite position at a line, interpolated between lattice rows."""
        return np.array(
            [np.interp(row, self.rows, self.sat_pos[:, i]) for i in range(3)]
        )

    @property
    def center_pixel(self):
        """The image center as a (line, sample) pair."""
        return self.image_rows / 2.0, self.image_cols / 2.0

    def footprint(self):
        """Scene footprint in EPSG:4326, traced around the image border.

        The lattice overshoots the image (a few hundred lines beyond the first
        and last), so the border is *interpolated* onto the ground lattice
        rather than taken from the lattice extent — using the lattice extent
        directly overstates the nadir band's area by roughly 14%.
        """
        line = np.linspace(0, self.image_rows, _FOOTPRINT_EDGE_SAMPLES)
        sample = np.linspace(0, self.image_cols, _FOOTPRINT_EDGE_SAMPLES)
        border = np.vstack(
            [
                np.c_[np.zeros_like(sample), sample],  # first line, left to right
                np.c_[line, np.full_like(line, self.image_cols)],  # last sample
                np.c_[np.full_like(sample, self.image_rows), sample[::-1]],  # last line
                np.c_[line[::-1], np.zeros_like(line)],  # first sample, back up
            ]
        )
        lon, lat = _ecef_to_lonlat(self._interp(border))
        return Polygon(zip(lon, lat))

    def view_angles(self):
        """Satellite view geometry at the scene center.

        Azimuth and elevation are measured at the ground point, in its local
        ENU frame, matching the DigitalGlobe ``MEANSATAZ``/``MEANSATEL``
        convention the pair math (convergence angle, BIE) expects. The
        off-nadir, in-track and cross-track angles are measured at the
        spacecraft, against geodetic nadir; in-track is positive when the
        instrument looks forward along the velocity vector and cross-track
        positive to starboard, so ASTER's backward-looking band 3B reports a
        negative in-track angle.

        In-track and cross-track are the look direction's projections onto the
        two planes (as in the vendor summaries this mirrors), so they combine
        to slightly more than the total off-nadir angle rather than exactly it.

        Returns
        -------
        dict
            ``meansataz``, ``meansatel``, ``meanoffnadirviewangle``,
            ``meanintrackviewangle`` and ``meancrosstrackviewangle``, in
            degrees.
        """
        row, col = self.center_pixel
        ground = self.ground_at(row, col)
        sat = self.sat_at(row)

        # Ground-based azimuth/elevation of the satellite.
        lon, lat = _ecef_to_lonlat(ground)
        east, north, up = _enu_basis(float(lon), float(lat))
        to_sat = sat - ground
        to_sat /= np.linalg.norm(to_sat)
        az = np.degrees(np.arctan2(to_sat @ east, to_sat @ north)) % 360.0
        el = np.degrees(np.arcsin(np.clip(to_sat @ up, -1.0, 1.0)))

        # Spacecraft-based off-nadir decomposition. The velocity direction is
        # differenced from the neighbouring lattice-row positions (the camera
        # file carries positions only), then orthogonalized against nadir.
        sat_lon, sat_lat = _ecef_to_lonlat(sat)
        _, _, sat_up = _enu_basis(float(sat_lon), float(sat_lat))
        down = -sat_up
        velocity = self.sat_pos[-1] - self.sat_pos[0]
        along = velocity - (velocity @ down) * down
        along /= np.linalg.norm(along)
        starboard = np.cross(down, along)

        look = -to_sat
        return {
            "meansataz": float(np.round(az, 2)),
            "meansatel": float(np.round(el, 2)),
            "meanoffnadirviewangle": float(
                np.round(np.degrees(np.arccos(np.clip(look @ down, -1.0, 1.0))), 2)
            ),
            "meanintrackviewangle": float(
                np.round(np.degrees(np.arctan2(look @ along, look @ down)), 2)
            ),
            "meancrosstrackviewangle": float(
                np.round(np.degrees(np.arctan2(look @ starboard, look @ down)), 2)
            ),
        }

    def gsd(self):
        """Mean ground sample distance (m) at the scene center.

        Averages the ground spacing per pixel in the sample and line
        directions, measured between the lattice cells straddling the center.
        """
        row, col = self.center_pixel
        d_col = self.cols[1] - self.cols[0]
        d_row = self.rows[1] - self.rows[0]
        across = np.linalg.norm(
            self.ground_at(row, col + d_col / 2) - self.ground_at(row, col - d_col / 2)
        )
        along = np.linalg.norm(
            self.ground_at(row + d_row / 2, col) - self.ground_at(row - d_row / 2, col)
        )
        return float(np.round((across / d_col + along / d_row) / 2.0, 2))

    def ephemeris(self):
        """Satellite positions as a GeoDataFrame indexed by lattice line.

        ASTER camera files carry no timestamps and no velocities, so this is
        the positions-only trajectory: indexed by image **line number** rather
        than time, with NaN covariance columns so consumers can test for
        "provided" the same way they do for every other sensor.
        """
        eph_df = pd.DataFrame(self.sat_pos, columns=["x", "y", "z"])
        for name in ("dx", "dy", "dz"):
            eph_df[name] = np.nan
        for name in ("11", "12", "13", "22", "23", "33"):
            eph_df[f"cov_{name}"] = np.nan
        eph_df["line"] = self.rows
        eph_df.set_index("line", inplace=True)
        return gpd.GeoDataFrame(
            eph_df,
            geometry=gpd.points_from_xy(eph_df["x"], eph_df["y"], eph_df["z"]),
            crs="EPSG:4978",
        )


class AsterMetadata(SensorMetadata):
    """Metadata reader for ASP ``gen_aster`` ASTER camera XMLs.

    One scene per band file, so a 3N/3B pair goes through the existing
    N-choose-2 pair machinery with no special-casing: the derived view angles
    reproduce ASTER's nadir/backward stereo geometry (~31° convergence at the
    ground for the nominal 27.6° backward pointing).

    See the module docstring for what is derived and what is unavailable. In
    short: footprint, view angles, GSD and a positions-only trajectory are
    derived; attitude does not exist (``att_df`` is None), sun angles and
    cloud cover are absent, and the date comes from a neighbouring L1A granule
    name if there is one.
    """

    name = "ASTER"

    def __init__(self, directory=None, image_list=None):
        """
        Initialize an ASTER metadata reader.

        Parameters
        ----------
        directory : str, optional
            Path to directory containing ``gen_aster`` camera XMLs.
        image_list : list of str, optional
            Explicit list of XML files to use instead of discovering them from
            ``directory``.

        Raises
        ------
        ValueError
            If neither ``directory`` nor ``image_list`` is given, or if no
            gen_aster camera files are found.
        """
        self.directory, self.image_list = resolve_input_files(
            type(self), directory, image_list
        )
        if not self.image_list:
            raise ValueError(
                "\n\nMissing ASTER gen_aster camera XML files. "
                "Cannot extract metadata without these.\n\n"
            )

    @classmethod
    def _is_camera_file(cls, path):
        """True if ``path`` is a gen_aster camera XML.

        The ``<isd>`` root alone is not enough — DigitalGlobe-heritage camera
        files use it too — so the lattice blocks are what this claims on, and
        symmetrically what keeps the WorldView reader from claiming these.
        """
        return _is_aster_isd(path)

    def get_scene_dicts(self):
        """Return one sensor-agnostic scene dict per camera XML."""
        return [self.get_scene_dict(xml) for xml in self.image_list]

    def get_scene_dict(self, xml, geteph=True):
        """
        Get a dictionary of metadata for one ASTER band.

        Parameters
        ----------
        xml : str
            Path to the ``gen_aster`` camera XML.
        geteph : bool, optional
            Whether to include the trajectory block, default is True.

        Returns
        -------
        dict
            Sensor-agnostic scene dict (see the package docstring).
        """
        root = ET.parse(xml).getroot()
        lattice = _AsterLattice(root)
        geom = lattice.footprint()

        d = {
            "xml_fn": xml,
            # gen_aster names its outputs <prefix>-Band3N.xml / -Band3B.xml,
            # and SATID is identical for both bands, so the file stem is what
            # distinguishes the scenes of a pair.
            "catid": os.path.splitext(os.path.basename(xml))[0],
            "sensor": root.findtext(".//SATID") or "ASTER",
            "date": _acquisition_date(xml),
            "geom": geom,
            "meanproductgsd": lattice.gsd(),
            **lattice.view_angles(),
        }

        if geteph:
            d["eph_gdf"] = lattice.ephemeris()
            # ASTER camera files carry no attitude at all; None is the
            # documented "not provided" value and consumers annotate the
            # orientation panels rather than plotting nothing.
            d["att_df"] = None
            d["fp_gdf"] = gpd.GeoDataFrame(
                {"idx": [0], "geometry": [geom]},
                geometry="geometry",
                crs="EPSG:4326",
            )

        return fill_scene_defaults(d)
