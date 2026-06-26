"""Reconstruct ``mapproject`` commands from ASP-mapprojected GeoTIFF metadata.

ASP's ``mapproject`` tool does not write a log file the way ``bundle_adjust`` /
``stereo`` / ``point2dem`` do, so :mod:`asp_plot.asp_log` (which parses those
logs) has nothing to read for the mapprojection step (issue #96).

Fortunately ``mapproject`` stamps everything we need into the *output* GeoTIFF
header, so the command can be reconstructed from the output data alone -- no new
ASP ``--log`` flag required. The fields written by ASP are:

- ``INPUT_IMAGE_FILE`` -- the image that was mapprojected
- ``CAMERA_FILE`` -- the camera model
- ``DEM_FILE`` -- the DEM used as the projection surface
- ``CAMERA_MODEL_TYPE`` -- the resolved ``--session-type`` / ``-t``
- ``BUNDLE_ADJUST_PREFIX`` -- the ``--bundle-adjust-prefix`` (``"NONE"`` if unset)

combined with the raster's own CRS (``--t_srs``), resolution (``--tr``), and
bounds (``--t_projwin``).

The reconstruction is faithful but *not* byte-for-byte re-runnable: the session
type is the resolved value (not necessarily what the user typed), an input
``--mpp`` shows up resolved as ``--tr``, and the output name is read from the
file itself. Callers that surface this to users (e.g. the PDF report) should say
so. See ``reconstruct_mapproject_command`` for the exact argv order.
"""

import glob
import logging
import os
import warnings

import rasterio
from rasterio.errors import NotGeoreferencedWarning

logger = logging.getLogger(__name__)

# GeoTIFF metadata tags ASP's mapproject writes into every output. This trio is
# the signature we require to treat a raster as a mapproject output -- and the
# *only* thing we rely on to identify mapprojected files. We deliberately do not
# match on filename conventions (``*_map.tif`` etc.): the tags are written by ASP
# itself, are self-validating, and survive a rename, whereas a filename glob
# would force an external naming convention and miss anything that deviates. All
# three are read back during reconstruction, so requiring them here also guards
# the ``tags[...]`` lookups below.
REQUIRED_TAGS = ("INPUT_IMAGE_FILE", "CAMERA_FILE", "DEM_FILE")

# Raster extensions scanned when discovering mapproject outputs in a directory.
RASTER_EXTENSIONS = ("*.tif", "*.tiff")


def _format_coord(value):
    """Format a projwin/resolution coordinate without scientific notation.

    GeoTIFF bounds can be large UTM northings (~5.2e6); ``%g`` would render those
    in exponential form, which is unusable as a CLI argument. We emit a plain
    decimal, trimming trailing zeros so integers stay integers.
    """
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def reconstruct_mapproject_command(raster_path):
    """Reconstruct the ``mapproject`` command for an ASP-mapprojected GeoTIFF.

    Parameters
    ----------
    raster_path : str
        Path to a GeoTIFF that may be an ASP ``mapproject`` output.

    Returns
    -------
    str or None
        The reconstructed ``mapproject ...`` command line, or ``None`` if the
        file is missing the ASP mapproject metadata signature (i.e. it is not a
        mapproject output, or was written by a tool that did not stamp the
        header).

    Notes
    -----
    The reconstructed argv order mirrors the ASP CLI::

        mapproject [-t SESSION] [--t_srs SRS] --tr TR \\
            --t_projwin XMIN YMIN XMAX YMAX \\
            [--bundle-adjust-prefix PREFIX] \\
            DEM_FILE INPUT_IMAGE CAMERA OUTPUT

    ``--t_srs`` is emitted as ``EPSG:XXXX`` when the CRS has an exact EPSG code,
    otherwise as the PROJ string (quoted), so custom planetary/local projections
    (e.g. the stereographic frames used in jitter solving) still round-trip.
    """
    try:
        # Discovery opens every candidate raster, including the raw (non-
        # georeferenced) input scenes; rasterio's NotGeoreferencedWarning on
        # those is expected and noise here, so silence it. The filter must be
        # set before open(), where the warning is emitted.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            ds = rasterio.open(raster_path)
        with ds:
            tags = ds.tags()
            if not all(tags.get(k) for k in REQUIRED_TAGS):
                return None

            crs = ds.crs
            res = ds.res
            bounds = ds.bounds

            parts = ["mapproject"]

            session = tags.get("CAMERA_MODEL_TYPE")
            if session:
                parts += ["-t", session]

            t_srs = _t_srs_token(crs)
            if t_srs:
                parts += ["--t_srs", t_srs]

            parts += ["--tr", _format_coord(res[0])]
            parts += [
                "--t_projwin",
                _format_coord(bounds.left),
                _format_coord(bounds.bottom),
                _format_coord(bounds.right),
                _format_coord(bounds.top),
            ]

            ba_prefix = tags.get("BUNDLE_ADJUST_PREFIX")
            if ba_prefix and ba_prefix != "NONE":
                parts += ["--bundle-adjust-prefix", ba_prefix]

            parts += [
                tags["DEM_FILE"],
                tags["INPUT_IMAGE_FILE"],
                tags["CAMERA_FILE"],
                os.path.basename(raster_path),
            ]
            return " ".join(parts)
    except rasterio.errors.RasterioIOError as e:
        logger.warning("Could not read %s for mapproject metadata: %s", raster_path, e)
        return None


def _t_srs_token(crs):
    """Return the ``--t_srs`` token for a rasterio CRS (or ``None``).

    Prefers a compact ``EPSG:XXXX`` when the CRS resolves to an exact EPSG code;
    falls back to the PROJ string (quoted, since it contains spaces) so custom
    projections without an EPSG code are still representable on the CLI.
    """
    if crs is None:
        return None
    epsg = crs.to_epsg()
    if epsg:
        return f"EPSG:{epsg}"
    proj4 = crs.to_proj4()
    if proj4:
        return f'"{proj4}"'
    return None


def find_mapproject_commands(directories, stereo_command=None):
    """Find ASP-mapprojected outputs across ``directories`` and reconstruct them.

    Scans each directory (non-recursively) for GeoTIFFs and keeps those carrying
    the ASP mapproject metadata signature (see ``REQUIRED_TAGS``) -- identity is
    decided entirely by the file's own metadata, not by its name, so the result
    is robust to ASP/asp_plot filename conventions. Each kept output is
    reconstructed into a command; results are deduplicated (the same scene can be
    reached via more than one of the passed directories) and returned sorted for
    stable report output.

    Parameters
    ----------
    directories : iterable of str or None
        Directories to scan (e.g. the processing root, stereo dir, BA dir).
        ``None`` entries are skipped.
    stereo_command : str, optional
        The stereo command line for the run being reported. When given, a
        mapprojected output is kept only if its filename appears among the
        stereo inputs (i.e. this run actually consumed it). This scopes the
        result to the run at hand: a non-mapprojected stereo run that shares a
        directory with leftover mapprojected scenes (e.g. ``stereo/`` and
        ``stereo_no_mapproj/`` under one parent) no longer picks them up. A
        mapprojected run's ``stereo`` command lists the ``*_map.tif`` inputs; a
        non-mapprojected run lists the raw images instead, so the gate is a
        plain filename-membership test -- no fragile positional parsing. When
        ``None`` (or empty), no gating is applied and every discovered output is
        returned.

    Returns
    -------
    list of str
        Reconstructed ``mapproject`` command lines, sorted; empty if none found.
    """
    seen = set()
    commands = []
    scanned = set()
    for directory in directories:
        if not directory:
            continue
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            continue
        candidates = set()
        for ext in RASTER_EXTENSIONS:
            candidates.update(glob.glob(os.path.join(directory, ext)))
        for path in sorted(candidates):
            real = os.path.realpath(path)
            if real in scanned:
                continue
            scanned.add(real)
            # Scope to what this stereo run actually used: a mapprojected output
            # is an input to mapprojected stereo, so its basename appears in the
            # stereo command. Skip outputs the run did not consume.
            if stereo_command and os.path.basename(path) not in stereo_command:
                continue
            command = reconstruct_mapproject_command(path)
            # Dedupe on the reconstructed command: distinct left/right scenes
            # differ (different input image + output name), but the same scene
            # copied into two scanned directories yields an identical command.
            if command and command not in seen:
                seen.add(command)
                commands.append(command)
    return sorted(commands)
