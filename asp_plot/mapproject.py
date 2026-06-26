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

import logging
import os

import rasterio

logger = logging.getLogger(__name__)

# GeoTIFF metadata tags ASP's mapproject writes into every output. The first two
# are the minimal signature we require to treat a raster as a mapproject output;
# without an input image and camera there is nothing meaningful to reconstruct.
REQUIRED_TAGS = ("INPUT_IMAGE_FILE", "CAMERA_FILE")

# Filename globs ASP/asp_plot conventionally use for mapprojected scenes. Used to
# narrow directory scans before reading headers; the tag check is authoritative.
MAPPROJECT_GLOBS = ("*_map.tif", "*_proj.tif", "*.map.tif", "*-mapproj.tif")


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
        with rasterio.open(raster_path) as ds:
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


def find_mapproject_commands(directories):
    """Find ASP-mapprojected outputs across ``directories`` and reconstruct them.

    Scans each directory (non-recursively) for GeoTIFFs matching the known
    mapproject filename conventions, keeps those carrying the ASP mapproject
    metadata signature, and reconstructs a command for each. Results are
    deduplicated by input image (a scene can be discovered under more than one
    of the passed directories) and returned sorted for stable report output.

    Parameters
    ----------
    directories : iterable of str or None
        Directories to scan (e.g. the processing root, stereo dir, BA dir).
        ``None`` entries are skipped.

    Returns
    -------
    list of str
        Reconstructed ``mapproject`` command lines, sorted; empty if none found.
    """
    import glob

    seen_inputs = set()
    commands = []
    scanned = set()
    for directory in directories:
        if not directory:
            continue
        directory = os.path.expanduser(directory)
        if not os.path.isdir(directory):
            continue
        candidates = set()
        for pattern in MAPPROJECT_GLOBS:
            candidates.update(glob.glob(os.path.join(directory, pattern)))
        for path in candidates:
            real = os.path.realpath(path)
            if real in scanned:
                continue
            scanned.add(real)
            command = reconstruct_mapproject_command(path)
            if not command:
                continue
            # Dedupe by (input image, output name) so the left/right pair both
            # show, but the same file reached via two directories does not.
            with rasterio.open(path) as ds:
                key = (ds.tags().get("INPUT_IMAGE_FILE"), os.path.basename(path))
            if key in seen_inputs:
                continue
            seen_inputs.add(key)
            commands.append(command)
    return sorted(commands)
