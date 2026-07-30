"""Sensor-specific metadata readers for stereo scenes.

This package isolates the *sensor-specific* work of discovering scene files and
extracting per-scene metadata from the *sensor-agnostic* stereo-pair geometry
math in :mod:`asp_plot.stereopair_metadata_parser`.

The goal is flexibility: WorldView (and other DigitalGlobe-heritage) XML camera
files, the Airbus DIMAP v2 family (Pléiades 1A/1B and Neo, SPOT 6/7, PeruSat-1),
the DIMAP v1 family (SPOT 5, ALOS PRISM), ASTER ``gen_aster`` camera files, and
RPC-only products (Cartosat-1, Deimos, anything ASP runs with ``-t rpc``) are
supported, and adding a new sensor is a matter of writing a new
:class:`SensorMetadata` subclass in its own module and registering it in
``SENSORS`` — no changes to the pair-level geometry code are required.

Most readers *parse* metadata a vendor wrote down; :mod:`asp_plot.sensors.aster`
and :mod:`asp_plot.sensors.rpc` instead *derive* their scene dicts (footprint,
view angles, GSD, satellite positions) from look vectors, because their camera
files record no summary geometry at all. Both kinds fill the same schema.

Most readers also claim *XML* files; the RPC reader claims images, because
that is where an RPC-only product keeps its camera model. It is therefore
marked ``fallback`` and consulted only after every XML reader has declined,
since a WorldView or Pléiades delivery ships images alongside its camera XMLs.

Each reader is responsible for turning a directory of camera/metadata files into
a list of *scene dicts*, one per scene, each containing the sensor-agnostic keys
the geometry code consumes:

``xml_fn``, ``catid``, ``sensor``, ``date``, ``scandir``, ``tdi``, ``geom``
(a Shapely polygon footprint in EPSG:4326), the mean view-angle/GSD/sun
attributes (``meansataz``, ``meansatel``, ``meanoffnadirviewangle``,
``meanintrackviewangle``, ``meancrosstrackviewangle``, ``meanproductgsd``,
``meansunaz``, ``meansunel``, ``cloudcover``), and — when ``geteph`` is True —
``eph_gdf`` (ephemeris GeoDataFrame in EPSG:4978), ``att_df`` (attitude
DataFrame), and ``fp_gdf`` (footprint GeoDataFrame in EPSG:4326).

``eph_gdf`` and ``fp_gdf`` are the trajectory block's own optional members:
``fp_gdf`` is always provided, while ``eph_gdf`` is **omitted entirely** by a
reader that can recover no satellite positions at all (an RPC-only product
whose look rays do not converge). Consumers use ``d.get("eph_gdf")`` and plot
the footprint alone when it is absent.

``att_df`` comes in one of two shapes, depending on what the sensor reports:
quaternions (``q1..q4``, scalar-last) or the vendor's own roll/pitch/yaw
(``roll``/``pitch``/``yaw`` in degrees, with ``attrs["rpy_frame"]`` naming the
frame they are defined in). Both carry NaN-filled ``cov_*`` columns when the
format has no covariance. Consumers dispatch on which columns are present, and
must tolerate ``att_df`` being None: a sensor whose camera files record no
attitude at all (ASTER) reports it that way rather than inventing one.

``eph_gdf`` is time-indexed for every sensor that timestamps its trajectory;
ASTER and RPC-only products, which timestamp nothing, index it by image line
number instead.

Layout: :mod:`asp_plot.sensors.base` holds the :class:`SensorMetadata` ABC and
shared helpers; each sensor family lives in its own module
(:mod:`asp_plot.sensors.worldview`, :mod:`asp_plot.sensors.dimap`,
:mod:`asp_plot.sensors.dimap_v1`, :mod:`asp_plot.sensors.aster`,
:mod:`asp_plot.sensors.rpc`); this
``__init__`` holds the ``SENSORS`` registry and the detection entry points, and
re-exports every public name so ``from asp_plot.sensors import ...`` is stable
across the package split.
"""

import glob
import logging
import os

from asp_plot.sensors.aster import AsterMetadata
from asp_plot.sensors.base import (
    SensorMetadata,
    _common_base,
    list_candidate_images,
    list_candidate_xmls,
)
from asp_plot.sensors.dimap import PleiadesMetadata
from asp_plot.sensors.dimap_v1 import PrismMetadata, Spot5Metadata
from asp_plot.sensors.rpc import RpcMetadata
from asp_plot.sensors.worldview import WorldViewMetadata

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

__all__ = [
    "SENSORS",
    "SensorMetadata",
    "WorldViewMetadata",
    "PleiadesMetadata",
    "Spot5Metadata",
    "PrismMetadata",
    "AsterMetadata",
    "RpcMetadata",
    "resolve_camera_inputs",
    "resolve_xml_inputs",
    "sensor_for_directory",
    "sensor_for_inputs",
]

# Registry of available sensor readers, in detection-priority order. The DIMAP
# and ASTER readers identify strictly (root tag plus profile/mission tags, or
# the gen_aster lattice blocks) while the WorldView reader claims any XML
# carrying the DG camera blocks, so WorldView is checked last of the XML
# readers. RpcMetadata is a ``fallback`` reader (it claims *images*, which
# every delivery has) and is consulted only after all of these decline.
SENSORS = [
    PleiadesMetadata,
    Spot5Metadata,
    PrismMetadata,
    AsterMetadata,
    WorldViewMetadata,
    RpcMetadata,
]


def resolve_xml_inputs(inputs, recursive=True):
    """Expand files, directories, and glob patterns into XML file paths.

    Lets a user point the tools at messy inputs without a fixed directory
    structure — e.g. ``geom_plot *.XML`` (already expanded by the shell),
    ``geom_plot scene1.xml scene2.xml``, ``geom_plot delivery_dir/``, or a mix.

    Each item of ``inputs`` may be:

    - a path to an XML file (included directly),
    - a directory (searched with the sensor-neutral
      :func:`asp_plot.sensors.base.list_candidate_xmls`, which is
      shallow-first and falls back to a recursive search), or
    - a glob pattern (expanded with :func:`glob.glob`).

    Results are de-duplicated (by absolute path) and returned sorted. Directory
    inputs get only the generic basename filter (``README.XML``, ortho
    products); the sensor-specific *content* checks are applied by the
    readers, not here.

    Parameters
    ----------
    inputs : str or os.PathLike or iterable of those
        One or more files, directories, and/or glob patterns.
    recursive : bool, optional
        Passed through to directory discovery and ``**`` glob expansion.
        Default True.

    Returns
    -------
    list of str
        Sorted, de-duplicated XML file paths.
    """
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]

    collected = []
    for item in inputs:
        item = os.path.expanduser(str(item))
        if os.path.isdir(item):
            collected.extend(list_candidate_xmls(item, recursive=recursive))
        elif glob.has_magic(item):
            collected.extend(glob.glob(item, recursive=recursive))
        elif os.path.isfile(item):
            collected.append(item)
        else:
            logger.warning("Input does not exist, skipping: %s", item)

    seen = set()
    unique = []
    for path in collected:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return sorted(unique)


def resolve_camera_inputs(inputs, recursive=True):
    """Expand inputs into every candidate camera file: XMLs *and* images.

    :func:`resolve_xml_inputs` covers the readers whose camera model is a
    sidecar XML; RPC-only products carry theirs inside the image (issue #177),
    so directory inputs also contribute candidate rasters here. Explicitly
    named files are included whatever their extension, so
    ``stereo_geom fore.tif aft.tif`` works.

    Mixing the two kinds in one list is safe: every reader filters the list
    through its own :meth:`SensorMetadata._is_camera_file` content check, so a
    WorldView delivery's images are ignored by the WorldView reader and its
    XMLs by the RPC reader.

    Parameters
    ----------
    inputs : str or os.PathLike or iterable of those
        One or more files, directories, and/or glob patterns.
    recursive : bool, optional
        Passed through to directory discovery and ``**`` glob expansion.
        Default True.

    Returns
    -------
    list of str
        Sorted, de-duplicated candidate camera file paths.
    """
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]
    inputs = list(inputs)

    collected = list(resolve_xml_inputs(inputs, recursive=recursive))
    seen = {os.path.abspath(p) for p in collected}
    for item in inputs:
        item = os.path.expanduser(str(item))
        if not os.path.isdir(item):
            continue
        for path in list_candidate_images(item, recursive=recursive):
            key = os.path.abspath(path)
            if key not in seen:
                seen.add(key)
                collected.append(path)
    return sorted(collected)


def sensor_for_inputs(inputs):
    """Detect and instantiate the appropriate sensor reader for explicit inputs.

    The file-list counterpart of :func:`sensor_for_directory`. Resolves
    ``inputs`` (files, directories, and/or globs) into a list of candidate
    camera files, then returns an instance of the first registered reader whose
    :meth:`SensorMetadata.detect_files` matches — ``fallback`` readers last, so
    a delivery's images cannot outrank its own camera XMLs.

    Parameters
    ----------
    inputs : str or os.PathLike or iterable of those
        One or more files, directories, and/or glob patterns.

    Returns
    -------
    SensorMetadata
        An initialized reader for the detected sensor.

    Raises
    ------
    ValueError
        If no candidate files are found, or no registered sensor reader
        matches them.
    """
    image_list = resolve_camera_inputs(inputs)
    if not image_list:
        raise ValueError(
            "\n\nNo camera metadata files found for the given input(s). "
            "Provide camera XML files, images carrying RPCs, a directory, or "
            "a glob pattern.\n\n"
        )
    base = _common_base(image_list)
    for sensor_cls in sorted(SENSORS, key=lambda s: s.fallback):
        if sensor_cls.detect_files(image_list):
            return sensor_cls(directory=base, image_list=image_list)
    raise ValueError(
        "\n\nNo supported sensor metadata files found among the given input(s). "
        f"Supported sensors: {', '.join(s.name for s in SENSORS)}.\n\n"
    )


def sensor_for_directory(directory):
    """Detect and instantiate the appropriate sensor reader for a directory.

    Iterates the ``SENSORS`` registry and returns an instance of the first
    reader whose :meth:`SensorMetadata.detect` matches the directory contents.

    Parameters
    ----------
    directory : str
        Path to directory containing camera/metadata files.

    Returns
    -------
    SensorMetadata
        An initialized reader for the detected sensor.

    Raises
    ------
    ValueError
        If no registered sensor reader matches the directory contents.
    """
    directory = os.path.expanduser(directory)
    # Two passes, mirroring the readers' shallow-first discovery: a sensor
    # whose metadata sits at the directory's top level wins over one that only
    # matches somewhere inside a nested delivery, regardless of registry order.
    # ``fallback`` readers sit outside that competition entirely and are tried
    # only after both passes have failed: a delivery whose camera XMLs are
    # nested must still be read by its own reader, not claimed by the RPC
    # reader just because the delivery's images happen to sit higher up.
    for fallback in (False, True):
        candidates = [s for s in SENSORS if s.fallback is fallback]
        for recursive in (False, True):
            for sensor_cls in candidates:
                if sensor_cls.detect(directory, recursive=recursive):
                    return sensor_cls(directory)
    raise ValueError(
        "\n\nNo supported sensor metadata files found in directory. "
        f"Supported sensors: {', '.join(s.name for s in SENSORS)}.\n\n"
    )
