"""
Reproducible "figure selections" for asp_plot reports.

When re-processing the same scene with different ASP parameters, the diagnostic
figures silently change *what they show* between runs: a fresh ICESat-2 request
returns a slightly different point set, the "best" profile track flips, the
best/worst agreement segments move, and the detailed-hillshade clip boxes are
re-selected from the (re-processed) intersection-error raster. That makes
before/after comparison impossible.

This module persists every non-deterministic selection a run made to a small
YAML sidecar next to the report, and reads it back so a later run can replay the
same choices. It deliberately imports nothing from ``report.py`` / ``fpdf`` so
it stays safe to use from notebooks.

See issue: https://github.com/uw-cryo/asp_plot/issues/121
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

import rasterio as rio
import rasterio.warp  # noqa: F401  (rio.warp not imported by `import rasterio`)
import yaml

logger = logging.getLogger(__name__)

SELECTIONS_SCHEMA_VERSION = 1

# Labels (low/medium/high intersection-error uncertainty) matching the
# magenta/cyan/orange rectangles drawn by StereoPlotter.plot_detailed_hillshade.
HILLSHADE_CLIP_LABELS = ["low", "medium", "high"]


@dataclass
class FigureSelections:
    """
    Container for the reproducible selections made while building a report.

    All nested values are plain JSON/YAML-serializable types (dicts, lists,
    numbers, strings) so the object round-trips cleanly through YAML.

    Attributes
    ----------
    asp_plot_version : str or None
        asp_plot version that wrote the file (informational).
    dem_filename : str or None
        DEM the selections were computed against (informational).
    map_crs : str or None
        Map CRS string used by the report (informational).
    detailed_hillshade : dict or None
        ``{"subset_km": float, "intersection_error_percentiles": [..],
        "dem_crs": "EPSG:XXXX", "clips": [{"label": str,
        "bbox": [xmin, ymin, xmax, ymax], "pixel_offset": [row, col]}, ...]}``.
        ``bbox`` is in ``dem_crs`` map coordinates (robust to a re-gridded DEM).
    icesat2 : dict or None
        ``{"request": {..}, "parquet_cache": {key: path}, "profile_track":
        {"rgt": int, "cycle": int, "spot": int}, "segments": {"best": {..},
        "worst": {..}}}``. Omitted (None) for planetary DEMs.
    """

    # schema_version is declared first so it serializes at the TOP of the YAML
    # (we dump with sort_keys=False, which preserves field order).
    schema_version: int = field(default=SELECTIONS_SCHEMA_VERSION)
    asp_plot_version: Optional[str] = None
    dem_filename: Optional[str] = None
    map_crs: Optional[str] = None
    detailed_hillshade: Optional[dict] = None
    icesat2: Optional[dict] = None

    def to_dict(self):
        """Return a plain dict suitable for YAML serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Build a FigureSelections from a parsed YAML/JSON dict."""
        if data is None:
            return cls()
        return cls(
            asp_plot_version=data.get("asp_plot_version"),
            dem_filename=data.get("dem_filename"),
            map_crs=data.get("map_crs"),
            detailed_hillshade=data.get("detailed_hillshade"),
            icesat2=data.get("icesat2"),
            schema_version=data.get("schema_version", SELECTIONS_SCHEMA_VERSION),
        )


def write_selections_yaml(path, selections):
    """
    Write a FigureSelections to a YAML file.

    Parameters
    ----------
    path : str
        Destination path (e.g. ``<report_stem>_figure_selections.yml``).
    selections : FigureSelections
        Selections to serialize.
    """
    with open(path, "w") as f:
        yaml.safe_dump(
            selections.to_dict(), f, default_flow_style=False, sort_keys=False
        )
    logger.info(f"Wrote figure selections to {path}")


def read_selections_yaml(path):
    """
    Read a FigureSelections from a YAML file.

    Parameters
    ----------
    path : str
        Path to a previously written selections file.

    Returns
    -------
    FigureSelections
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    selections = FigureSelections.from_dict(data)
    if selections.schema_version != SELECTIONS_SCHEMA_VERSION:
        logger.warning(
            f"Figure selections schema version {selections.schema_version} "
            f"differs from supported {SELECTIONS_SCHEMA_VERSION}; "
            "attempting to use anyway."
        )
    return selections


# ---------------------------------------------------------------------------
# Geometry helpers: clip box <-> pixel window
#
# A detailed-hillshade clip is stored as a map-coordinate bounding box in the
# DEM's CRS so that reuse survives a re-gridded DEM (slightly different origin /
# size). On reuse we convert the bbox back to a top-left pixel offset in the
# *current* DEM.
# ---------------------------------------------------------------------------


def pixel_window_to_bbox(transform, row, col, n_rows, n_cols):
    """
    Convert a pixel window (top-left row/col + size) to a map-coordinate bbox.

    Parameters
    ----------
    transform : affine.Affine
        Raster geotransform (``raster.ds.transform``).
    row, col : int
        Top-left pixel of the window.
    n_rows, n_cols : int
        Window height/width in pixels.

    Returns
    -------
    list of float
        ``[xmin, ymin, xmax, ymax]`` in the raster's CRS.
    """
    ul_x, ul_y = rio.transform.xy(transform, row, col, offset="ul")
    lr_x, lr_y = rio.transform.xy(transform, row + n_rows, col + n_cols, offset="ul")
    xmin, xmax = sorted([ul_x, lr_x])
    ymin, ymax = sorted([ul_y, lr_y])
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


def reproject_bbox(bbox, src_crs, dst_crs):
    """
    Reproject a map-coordinate bbox from one CRS to another.

    Used when replaying detailed-hillshade clips against a DEM in a different
    CRS than the run that wrote them (e.g. a mapprojected vs. non-mapprojected
    stereo variant of the same scene, which can land in different projections).
    Returns the input unchanged when either CRS is missing or they are equal.

    Parameters
    ----------
    bbox : sequence of float
        ``[xmin, ymin, xmax, ymax]`` in ``src_crs``.
    src_crs : str or rasterio.crs.CRS or None
        CRS the bbox is currently expressed in.
    dst_crs : str or rasterio.crs.CRS or None
        Target CRS (the DEM being clipped on reuse).

    Returns
    -------
    list of float
        ``[xmin, ymin, xmax, ymax]`` in ``dst_crs``.
    """
    if not src_crs or not dst_crs:
        return list(bbox)
    src = rio.crs.CRS.from_user_input(src_crs)
    dst = rio.crs.CRS.from_user_input(dst_crs)
    if src == dst:
        return list(bbox)
    xmin, ymin, xmax, ymax = rio.warp.transform_bounds(
        src, dst, bbox[0], bbox[1], bbox[2], bbox[3]
    )
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


def bbox_to_pixel_offset(transform, bbox):
    """
    Convert a map-coordinate bbox to a top-left pixel offset (row, col).

    The window size is *not* returned: on reuse the subset size is recomputed
    from ``subset_km`` and the current DEM's GSD so the ground footprint stays
    constant even if the DEM resolution changed. Only the top-left anchor is
    needed.

    Parameters
    ----------
    transform : affine.Affine
        Geotransform of the DEM being clipped on reuse.
    bbox : sequence of float
        ``[xmin, ymin, xmax, ymax]`` in the DEM's CRS.

    Returns
    -------
    tuple of int
        ``(row, col)`` top-left pixel offset (clamped to be non-negative).
    """
    # Upper-left corner in map space is (xmin, ymax) for north-up rasters.
    xmin, ymax = bbox[0], bbox[3]
    # Round to nearest (not floor): the bbox corner sits exactly on a pixel
    # boundary, so tiny floating-point error must not shift the offset by one
    # pixel — otherwise a replayed clip drifts by a pixel vs. the original run.
    row, col = rio.transform.rowcol(transform, xmin, ymax, op=round)
    return max(int(row), 0), max(int(col), 0)
