"""Abstract base class and shared helpers for sensor metadata readers.

See the :mod:`asp_plot.sensors` package docstring for an overview. This module
defines the two contracts every reader participates in:

**File detection** is content-based: each reader supplies a
:meth:`SensorMetadata._is_camera_file` predicate that inspects a candidate
file (cheaply, e.g. with ``iterparse`` stopping early) and claims only files
its sensor format actually produced. Discovery, filtering, ``detect``, and
``detect_files`` are implemented here once, on top of that predicate, so a
reader never claims another sensor's files just because they end in ``.xml``
(issue #162).

**The scene-dict schema** splits into a required core and optional blocks:

- *identity core* (required, a reader must raise if it cannot fill these):
  ``xml_fn``, ``catid``, ``sensor``, ``date``, ``geom``
- *summary block* (optional): the mean view-angle/GSD/sun fields and
  ``cloudcover`` default to NaN; ``scandir`` and ``tdi`` default to None
- *trajectory block* (optional, when ``geteph`` is True): ``eph_gdf``,
  ``att_df``, ``fp_gdf``

Readers call :func:`fill_scene_defaults` so a format that lacks a field
degrades to the documented "not provided" value instead of omitting the key,
and consumers rely on one convention (None/NaN render as omitted/"nan"/"N/A")
rather than per-key accidents (issue #163).

Two small geodesy helpers live here as well (:func:`_ecef_to_lonlat`,
:func:`_lonlat_to_ecef` and :func:`_enu_basis`), because the two
derived-geometry readers — :mod:`asp_plot.sensors.aster` and
:mod:`asp_plot.sensors.rpc` — both need to express a look direction in the
local east/north/up frame at a ground point.
"""

import os
import re
from abc import ABC, abstractmethod

import numpy as np
from pyproj import Transformer

from asp_plot.utils import glob_file

# XML files that are delivered alongside camera models but are *not* camera
# models themselves and are excluded by name before any content check: ortho
# products (``*ortho*.xml``) and the ``README.XML`` that ships in every
# DigitalGlobe-heritage delivery. Matched against the file basename,
# case-insensitively.
_NON_CAMERA_XML_RE = re.compile(r"ortho|readme", re.IGNORECASE)

#: Raster extensions searched when a reader's "camera file" is the image
#: itself rather than a sidecar XML (RPC-only products, issue #177). Matched
#: case-insensitively against the file extension.
IMAGE_EXTENSIONS = (".ntf", ".nitf", ".tif", ".tiff", ".jp2")

_ECEF_TO_LONLAT = None
_LONLAT_TO_ECEF = None


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


def _ecef_height(xyz):
    """Geodetic height (m) above the WGS84 ellipsoid of an ECEF position."""
    global _ECEF_TO_LONLAT
    if _ECEF_TO_LONLAT is None:
        _ECEF_TO_LONLAT = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    x, y, z = np.asarray(xyz, dtype=float)
    return float(_ECEF_TO_LONLAT.transform(x, y, z)[2])


def _lonlat_to_ecef(lon, lat, height):
    """Convert geodetic (lon, lat, height) to ECEF x, y, z in metres.

    Parameters
    ----------
    lon, lat : array_like
        Geodetic longitude and latitude in degrees.
    height : array_like
        Height above the WGS84 ellipsoid in metres.

    Returns
    -------
    numpy.ndarray
        ECEF coordinates, shape ``(..., 3)``.
    """
    global _LONLAT_TO_ECEF
    if _LONLAT_TO_ECEF is None:
        _LONLAT_TO_ECEF = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    x, y, z = _LONLAT_TO_ECEF.transform(lon, lat, height)
    return np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=-1)


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


#: Optional scene-dict fields and their "not provided" values. The NaN
#: defaults are floats so downstream ``%0.1f``-style formatting renders "nan"
#: instead of raising; the None defaults are omitted from scene strings.
OPTIONAL_SCENE_FIELDS = {
    "scandir": None,
    "tdi": None,
    "meansataz": np.nan,
    "meansatel": np.nan,
    "meanoffnadirviewangle": np.nan,
    "meanintrackviewangle": np.nan,
    "meancrosstrackviewangle": np.nan,
    "meanproductgsd": np.nan,
    "meansunaz": np.nan,
    "meansunel": np.nan,
    "cloudcover": np.nan,
}


def fill_scene_defaults(scene_dict):
    """Fill missing optional scene-dict fields with "not provided" values.

    Mutates and returns ``scene_dict``. Fields already present (even as
    None/NaN) are left alone; the required identity-core fields are not
    checked here — a reader that cannot fill those must raise instead.
    """
    for key, default in OPTIONAL_SCENE_FIELDS.items():
        scene_dict.setdefault(key, default)
    return scene_dict


def _glob_xmls(directory, recursive=False):
    """Glob ``*.xml``/``*.XML`` in ``directory`` (one level, or recursively)."""
    pattern = "**/*.[Xx][Mm][Ll]" if recursive else "*.[Xx][Mm][Ll]"
    # glob_file returns None (not []) when nothing matches. quiet: every
    # reader probes every directory during detection, so "no XMLs here" is a
    # normal answer, not a missing-plot warning.
    return (
        glob_file(directory, pattern, all_files=True, recursive=recursive, quiet=True)
        or []
    )


def list_candidate_xmls(directory, recursive=True):
    """List camera-*candidate* XMLs in ``directory``, name-filtered only.

    Searches the top level of ``directory`` first: a flat delivery or an ASP
    processing directory keeps its camera XMLs there, so those take precedence
    and unrelated XMLs in subdirectories are not pulled in. Only when the top
    level has no candidate XML does it fall back to a recursive search, because
    some satellite deliveries nest the camera XML several subdirectories deep
    (e.g. ``.../<order>/DVD_VOL_1/<order>/<scene>_PAN/<scene>.XML``).

    Non-camera XMLs (``*ortho*.xml``, ``README.XML``) are excluded by
    basename; no sensor-specific content check is applied — that is the
    readers' job (see :meth:`SensorMetadata._is_camera_file`). Used by
    :func:`asp_plot.sensors.resolve_xml_inputs` to expand directory inputs
    for *any* sensor.
    """
    found = _name_filter(_glob_xmls(directory))
    if not found and recursive:
        found = _name_filter(_glob_xmls(directory, recursive=True))
    return found


def _name_filter(image_list):
    """Drop non-camera XMLs (ortho products, README.XML, ...) by basename.

    Matched against the basename so a parent directory named e.g.
    ``ortho_run/`` does not exclude otherwise-valid scenes.
    """
    return sorted(
        f
        for f in (image_list or [])
        if not _NON_CAMERA_XML_RE.search(os.path.basename(f))
    )


def _glob_images(directory, recursive=False):
    """Glob raster files with an :data:`IMAGE_EXTENSIONS` extension.

    One ``*`` glob filtered by extension rather than one glob per extension:
    :func:`asp_plot.utils.glob_file` returns the first *pattern* that matches
    rather than the union of all of them.
    """
    pattern = "**/*" if recursive else "*"
    found = (
        glob_file(directory, pattern, all_files=True, recursive=recursive, quiet=True)
        or []
    )
    return sorted(
        f
        for f in found
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS and os.path.isfile(f)
    )


def list_candidate_images(directory, recursive=True):
    """List candidate camera *images* in ``directory``, by extension only.

    The image counterpart of :func:`list_candidate_xmls`, for products whose
    camera model is embedded in the image rather than delivered as a sidecar
    XML (RPC-only products, issue #177). Shallow-first with a recursive
    fallback for the same reason: a flat delivery or an ASP processing
    directory keeps its images at the top level.

    No content check is applied here — deciding whether an image actually
    carries RPCs is the reader's job (see
    :meth:`SensorMetadata._is_camera_file`), and it costs a raster header read
    per file.
    """
    found = _glob_images(directory)
    if not found and recursive:
        found = _glob_images(directory, recursive=True)
    return found


class SensorMetadata(ABC):
    """Abstract base class for a single sensor's metadata reader.

    A concrete reader discovers the scene files for one sensor in a directory and
    extracts a list of sensor-agnostic *scene dicts* (see the package docstring
    for the schema) that the stereo-pair geometry code can consume without
    knowing which sensor produced them.

    Subclasses must implement :meth:`_is_camera_file` (the content check that
    decides which files the sensor claims; discovery and detection are built
    on it here) and :meth:`get_scene_dicts`.

    Attributes
    ----------
    name : str
        Human-readable sensor name (e.g. ``"WorldView"``).
    fallback : bool
        Whether this reader is a last-resort match, tried only after every
        other reader has failed at *every* search depth. Set on readers that
        claim files a more specific reader's delivery also contains — the
        RPC reader claims images, and every WorldView or Pléiades delivery
        ships images next to its camera XMLs (issue #177).
    directory : str
        Path to the directory containing the sensor's metadata files.
    """

    name = "sensor"
    fallback = False

    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing the sensor's camera/metadata files.
        """
        self.directory = os.path.expanduser(directory)

    @classmethod
    @abstractmethod
    def _is_camera_file(cls, path):
        """Return True if ``path`` is a camera/metadata file of this sensor.

        The per-sensor content check behind discovery and detection. Must be
        cheap (many files may be inspected during a recursive delivery scan —
        e.g. ``iterparse`` stopping at the first identifying tags) and must
        reject other sensors' files, so unrecognized inputs produce a clean
        "no supported sensor found" error instead of a deep parse failure.
        """
        raise NotImplementedError

    @classmethod
    def _filter_camera_files(cls, image_list):
        """Keep only the files this sensor's content check claims, sorted."""
        return sorted(f for f in (image_list or []) if cls._is_camera_file(f))

    @classmethod
    def _discover_camera_files(cls, directory, recursive=True):
        """Discover this sensor's camera files in ``directory``.

        Shallow-first with a recursive fallback, mirroring
        :func:`list_candidate_xmls` but filtered by the sensor's own
        :meth:`_is_camera_file` content check: a flat processing directory
        keeps its camera XMLs at the top level, and only when none are found
        there does discovery descend into nested deliveries.

        Every reader in the package but one searches ``*.xml``; the RPC reader
        overrides this to search images instead (issue #177), which is why the
        method is named for camera *files* rather than XMLs.
        """
        found = cls._filter_camera_files(_glob_xmls(directory))
        if not found and recursive:
            found = cls._filter_camera_files(_glob_xmls(directory, recursive=True))
        return found

    @classmethod
    def detect(cls, directory, recursive=True):
        """Return True if this reader can handle the files in ``directory``.

        Parameters
        ----------
        directory : str
            Path to directory to inspect.
        recursive : bool, optional
            If True (default), also match metadata files nested in
            subdirectories. :func:`asp_plot.sensors.sensor_for_directory` first
            asks every sensor to detect shallowly and only then recursively, so
            a sensor matching at the top level wins over one matching a nested
            delivery.

        Returns
        -------
        bool
            Whether this sensor's metadata files are present.
        """
        return bool(
            cls._discover_camera_files(
                os.path.expanduser(directory), recursive=recursive
            )
        )

    @classmethod
    def detect_files(cls, image_list):
        """Return True if this reader can handle the files in ``image_list``.

        The file-list counterpart of :meth:`detect`, used when the sensor must
        be chosen from an explicit list of inputs rather than a directory.

        Parameters
        ----------
        image_list : list of str
            Candidate metadata file paths.

        Returns
        -------
        bool
            Whether this sensor's metadata files are present in the list.
        """
        return bool(cls._filter_camera_files(image_list))

    @abstractmethod
    def get_scene_dicts(self):
        """Return a list of per-scene metadata dictionaries.

        Returns
        -------
        list of dict
            One sensor-agnostic scene dict per scene (see package docstring).
        """
        raise NotImplementedError


def resolve_input_files(sensor_cls, directory, image_list):
    """Resolve a reader's ``(directory, image_list)`` constructor arguments.

    Every reader can be built either from a ``directory`` (its camera files are
    discovered) or from an explicit ``image_list`` (e.g. a shell-expanded
    ``stereo_geom *.XML``). Both paths end in the same place: a list filtered by
    the sensor's own :meth:`SensorMetadata._is_camera_file` content check, plus
    a directory to use for outputs and pair naming.

    Returns the resolved pair without raising on an empty result — each reader
    raises its own "missing files" error naming its format.

    Parameters
    ----------
    sensor_cls : type
        The :class:`SensorMetadata` subclass being constructed.
    directory : str or None
        Directory to discover files in, or the base directory for an explicit
        ``image_list``.
    image_list : list of str or None
        Explicit candidate files.

    Returns
    -------
    tuple of (str, list of str)
        The reader's ``directory`` and its filtered file list.

    Raises
    ------
    ValueError
        If neither ``directory`` nor ``image_list`` is given.
    """
    if directory is None and image_list is None:
        raise ValueError("Provide either a directory or an image_list.")

    if image_list is not None:
        files = sensor_cls._filter_camera_files(image_list)
        base = os.path.expanduser(directory) if directory else _common_base(files)
        return base, files

    base = os.path.expanduser(directory)
    return base, sensor_cls._discover_camera_files(base)


def _common_base(paths):
    """Return a base directory for a list of files.

    Used to pick a working directory (for ``dg_mosaic`` outputs and pair
    naming) when a reader is built from an explicit file list rather than a
    directory. Returns the files' common parent directory, or the current
    working directory if they share no common parent or the list is empty.
    """
    paths = [os.path.abspath(p) for p in (paths or [])]
    if not paths:
        return os.getcwd()
    base = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
    # commonpath can return a file path if one entry is a prefix of another;
    # make sure we hand back a directory.
    return base if os.path.isdir(base) else os.path.dirname(base)
