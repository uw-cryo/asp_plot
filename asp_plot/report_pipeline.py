"""Declarative report pipeline for the ``asp_plot`` CLI.

This module decouples the report-generation pipeline from Click. The CLI
(:mod:`asp_plot.cli.asp_plot`) parses options, packs them into a
:class:`ReportConfig`, and calls :func:`run_report` -- which is importable and
callable from notebooks and tests without any Click context.

The body of the report is a **declarative section registry**
(:data:`REPORT_SECTIONS`): each entry is a :class:`ReportSpec` pairing an
``enabled(ctx)`` predicate with a ``build(ctx)`` function that returns the
report sections to append. The orchestrator iterates the registry in order,
so section ordering and figure numbering are data, not source-line position.
Captions live in :mod:`asp_plot.report_captions`.
"""

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from typing import Callable, Iterator, List, Optional

import contextily as ctx

from asp_plot import report_captions as captions
from asp_plot.altimetry import Altimetry
from asp_plot.bodies import BODIES
from asp_plot.bundle_adjust import PlotBundleAdjustFiles, ReadBundleAdjustFiles
from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.report import (
    AlignmentReportPage,
    ReportMetadata,
    ReportSection,
    compile_report,
)
from asp_plot.scenes import ScenePlotter
from asp_plot.selections import (
    FigureSelections,
    read_selections_yaml,
    write_selections_yaml,
)
from asp_plot.stereo import StereoPlotter
from asp_plot.stereo_geometry import StereoGeometryPlotter
from asp_plot.utils import Raster, detect_planetary_body, get_acquisition_dates


@dataclass
class ReportConfig:
    """All inputs needed to generate a report, decoupled from Click.

    Field names and defaults mirror the CLI options one-for-one so the Click
    wrapper can splat its parsed params straight in.
    """

    directory: str = "./"
    bundle_adjust_directory: Optional[str] = None
    stereo_directory: str = "stereo"
    dem_filename: Optional[str] = None
    dem_gsd: Optional[float] = None
    map_crs: Optional[str] = None
    reference_dem: Optional[str] = None
    add_basemap: bool = True
    plot_altimetry: bool = True
    plot_icesat: Optional[object] = None
    altimetry_csv: Optional[str] = None
    pc_align: bool = True
    plot_geometry: bool = True
    subset_km: float = 1.0
    atl06sr_time_range: str = "all"
    reuse_selections: Optional[str] = None
    report_filename: Optional[str] = None
    report_title: Optional[str] = None
    # Reconstructed "asp_plot --flag ..." string recorded in the report. The
    # CLI builds this from the Click context; non-CLI callers may leave it None.
    report_command: Optional[str] = None


@dataclass
class ReportContext:
    """Mutable runtime state shared across section builders.

    Built once by :func:`_setup_context`, then threaded through every
    :class:`ReportSpec` so builders can pull shared plotters/selections and
    draw sequential figure filenames without a module-global counter.
    """

    config: ReportConfig
    plots_directory: str
    report_pdf_path: str
    report_title: str
    map_crs: Optional[str]
    ctx_kwargs: dict
    stereo_plotter: StereoPlotter
    asp_dem: Optional[str]
    plot_altimetry: bool
    report_metadata: Optional[ReportMetadata] = None
    reuse_clip_windows: Optional[list] = None
    reuse_clip_windows_crs: Optional[str] = None
    reuse_track: dict = field(default_factory=dict)
    reuse_segments: Optional[object] = None
    reuse_parquet: Optional[object] = None
    # ICESat-2 selections actually used, captured for the figure-selections file
    icesat2_selections: Optional[dict] = None
    _figure_counter: Iterator[int] = field(default_factory=lambda: count(0))

    def next_fig_fn(self) -> str:
        """Next sequential ``NN.png`` figure filename (00, 01, 02, ...)."""
        return f"{next(self._figure_counter):02}.png"

    def fig_path(self, fig_fn: str) -> str:
        return os.path.join(self.plots_directory, fig_fn)


@dataclass(frozen=True)
class ReportSpec:
    """One declarative report section.

    ``enabled(ctx)`` gates the section on config/runtime state; ``build(ctx)``
    runs the plotting and returns the :class:`ReportSection` /
    :class:`AlignmentReportPage` objects to append (possibly several, possibly
    none).
    """

    name: str
    enabled: Callable[["ReportContext"], bool]
    build: Callable[["ReportContext"], List[object]]


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_input_scenes(ctx: ReportContext) -> List[object]:
    cfg = ctx.config
    fig_fn = ctx.next_fig_fn()
    scene_plotter = ScenePlotter(
        cfg.directory, cfg.stereo_directory, title="Input Scenes"
    )
    scene_plotter.plot_scenes(save_dir=ctx.plots_directory, fig_fn=fig_fn)
    return [
        ReportSection(
            title="Input Scenes",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.INPUT_SCENES,
        )
    ]


def _build_stereo_geometry(ctx: ReportContext) -> List[object]:
    fig_fn = ctx.next_fig_fn()
    geom_plotter = StereoGeometryPlotter(
        ctx.config.directory, add_basemap=ctx.config.add_basemap
    )
    geom_plotter.dg_geom_plot(save_dir=ctx.plots_directory, fig_fn=fig_fn)
    return [
        ReportSection(
            title="Stereo Geometry",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.STEREO_GEOMETRY,
        )
    ]


def _build_match_points(ctx: ReportContext) -> List[object]:
    fig_fn = ctx.next_fig_fn()
    ctx.stereo_plotter.title = "Stereo Match Points"
    ctx.stereo_plotter.plot_match_points(save_dir=ctx.plots_directory, fig_fn=fig_fn)
    return [
        ReportSection(
            title="Match Points",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.MATCH_POINTS,
        )
    ]


def _build_bundle_adjust(ctx: ReportContext) -> List[object]:
    cfg = ctx.config
    sections: List[object] = []
    try:
        ba_files = ReadBundleAdjustFiles(cfg.directory, cfg.bundle_adjust_directory)
        resid_initial_gdf, resid_final_gdf = ba_files.get_initial_final_residuals_gdfs()

        plotter = PlotBundleAdjustFiles(
            [resid_initial_gdf, resid_final_gdf],
            lognorm=True,
            title="Bundle Adjust Initial and Final Residuals (Log Scale)",
        )

        fig_fn = ctx.next_fig_fn()
        plotter.plot_n_gdfs(
            column_name="mean_residual",
            cbar_label="Mean residual (px)",
            map_crs=ctx.map_crs,
            save_dir=ctx.plots_directory,
            fig_fn=fig_fn,
            **ctx.ctx_kwargs,
        )
        sections.append(
            ReportSection(
                title="Bundle Adjust Residuals (Log Scale)",
                image_path=ctx.fig_path(fig_fn),
                caption=captions.BUNDLE_RESIDUALS_LOG,
            )
        )

        plotter.lognorm = False
        plotter.title = "Bundle Adjust Initial and Final Residuals (Linear Scale)"

        fig_fn = ctx.next_fig_fn()
        plotter.plot_n_gdfs(
            column_name="mean_residual",
            cbar_label="Mean residual (px)",
            common_clim=False,
            map_crs=ctx.map_crs,
            save_dir=ctx.plots_directory,
            fig_fn=fig_fn,
            **ctx.ctx_kwargs,
        )
        sections.append(
            ReportSection(
                title="Bundle Adjust Residuals (Linear Scale)",
                image_path=ctx.fig_path(fig_fn),
                caption=captions.BUNDLE_RESIDUALS_LINEAR,
            )
        )

        # Map-projected residuals (requires reference DEM in bundle_adjust)
        try:
            resid_mapprojected_gdf = ba_files.get_mapproj_residuals_gdf()

            plotter = PlotBundleAdjustFiles(
                [resid_mapprojected_gdf],
                title="Bundle Adjust Midpoint distance between\nfinal interest points projected onto reference DEM",
            )

            fig_fn = ctx.next_fig_fn()
            plotter.plot_n_gdfs(
                column_name="mapproj_ip_dist_meters",
                cbar_label="Interest point distance (m)",
                map_crs=ctx.map_crs,
                save_dir=ctx.plots_directory,
                fig_fn=fig_fn,
                **ctx.ctx_kwargs,
            )
            sections.append(
                ReportSection(
                    title="Map-Projected Residuals",
                    image_path=ctx.fig_path(fig_fn),
                    caption=captions.MAP_PROJECTED_RESIDUALS,
                )
            )
        except ValueError as e:
            print(f"\n\nSkipping map-projected residuals plot: {e}\n\n")

        # Geodiff plots (requires reference DEM in bundle_adjust with --mapproj-dem flag)
        try:
            geodiff_initial_gdf, geodiff_final_gdf = (
                ba_files.get_initial_final_geodiff_gdfs()
            )

            plotter = PlotBundleAdjustFiles(
                [geodiff_initial_gdf, geodiff_final_gdf],
                lognorm=False,
                title="Bundle Adjust Initial and Final Geodiff vs. Reference DEM",
            )

            fig_fn = ctx.next_fig_fn()
            plotter.plot_n_gdfs(
                column_name="height_diff_meters",
                cbar_label="Height difference (m)",
                map_crs=ctx.map_crs,
                cmap="RdBu",
                symm_clim=True,
                save_dir=ctx.plots_directory,
                fig_fn=fig_fn,
                **ctx.ctx_kwargs,
            )
            sections.append(
                ReportSection(
                    title="Geodiff vs. Reference DEM",
                    image_path=ctx.fig_path(fig_fn),
                    caption=captions.GEODIFF,
                )
            )
        except ValueError as e:
            print(
                f"\n\nSkipping geodiff plots (requires --mapproj-dem flag in bundle_adjust): {e}\n\n"
            )

    except ValueError:
        print(
            f"\n\nNo bundle adjustment files found in directory {os.path.join(cfg.directory, cfg.bundle_adjust_directory):}. If you want bundle adjustment plots, make sure you run the tool and supply the correct directory to asp_plot.\n\n"
        )
    return sections


def _build_disparity(ctx: ReportContext) -> List[object]:
    fig_fn = ctx.next_fig_fn()
    ctx.stereo_plotter.title = "Disparity (pixels)"
    ctx.stereo_plotter.plot_disparity(
        unit="pixels", quiver=True, save_dir=ctx.plots_directory, fig_fn=fig_fn
    )
    return [
        ReportSection(
            title="Disparity",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.DISPARITY,
        )
    ]


def _build_dem_results(ctx: ReportContext) -> List[object]:
    fig_fn = ctx.next_fig_fn()
    ctx.stereo_plotter.title = "Stereo DEM Results"
    ctx.stereo_plotter.plot_dem_results(save_dir=ctx.plots_directory, fig_fn=fig_fn)
    return [
        ReportSection(
            title="DEM Results",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.DEM_RESULTS,
        )
    ]


def _build_detailed_hillshade(ctx: ReportContext) -> List[object]:
    fig_fn = ctx.next_fig_fn()
    ctx.stereo_plotter.title = "Hillshade with details"
    ctx.stereo_plotter.plot_detailed_hillshade(
        subset_km=ctx.config.subset_km,
        clip_windows=ctx.reuse_clip_windows,
        clip_windows_crs=ctx.reuse_clip_windows_crs,
        save_dir=ctx.plots_directory,
        fig_fn=fig_fn,
    )
    return [
        ReportSection(
            title="Detailed Hillshade",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.DETAILED_HILLSHADE,
        )
    ]


def _stats_row_from_result(align_result) -> dict:
    """First row of the alignment report dataframe, minus the 'key' column."""
    stats_row: dict = {}
    if (
        align_result.alignment_report_df is not None
        and not align_result.alignment_report_df.empty
    ):
        row = align_result.alignment_report_df.iloc[0].to_dict()
        row.pop("key", None)
        stats_row = row
    return stats_row


def _build_altimetry(ctx: ReportContext) -> List[object]:
    asp_dem = ctx.asp_dem

    # Auto-detect planetary body from DEM CRS
    body = detect_planetary_body(asp_dem) if asp_dem else "earth"
    print(f"\nDetected planetary body: {body}\n")

    # Auto-disable basemaps for non-Earth bodies
    if body != "earth":
        ctx_kwargs_altimetry = {}
    else:
        ctx_kwargs_altimetry = ctx.ctx_kwargs

    if body == "earth":
        return _build_altimetry_earth(ctx, ctx_kwargs_altimetry)
    elif body in ("moon", "mars"):
        return _build_altimetry_planetary(ctx, body)
    return []


def _build_altimetry_earth(ctx: ReportContext, ctx_kwargs_altimetry: dict):
    cfg = ctx.config
    asp_dem = ctx.asp_dem
    sections: List[object] = []

    # Parse --atl06sr_time_range into time_range/t0/t1 kwargs
    atl06sr_time_range = cfg.atl06sr_time_range
    atl06sr_time_kwargs = {}
    if atl06sr_time_range.lower() == "all":
        atl06sr_time_kwargs["time_range"] = "all"
    elif atl06sr_time_range.lower() == "auto":
        atl06sr_time_kwargs["time_range"] = "buffered"
    elif "," in atl06sr_time_range:
        parts = atl06sr_time_range.split(",", 1)
        atl06sr_time_kwargs["time_range"] = "buffered"
        atl06sr_time_kwargs["t0"] = parts[0].strip()
        atl06sr_time_kwargs["t1"] = parts[1].strip()
    else:
        # Single date → buffer around it
        atl06sr_time_kwargs["time_range"] = "buffered"
        atl06sr_time_kwargs["scene_date"] = atl06sr_time_range.strip()

    # Existing ICESat-2 workflow (3 plots: map, histogram, profile)
    icesat = Altimetry(directory=cfg.directory, dem_fn=asp_dem)

    # Reuse the exact prior points from parquet when replaying a prior
    # run's selections; otherwise request fresh from SlideRule (#121).
    loaded_from_reuse = False
    if ctx.reuse_parquet:
        loaded_from_reuse = icesat.load_atl06sr_from_parquet(ctx.reuse_parquet)
        if not loaded_from_reuse:
            print(
                "\nCould not reuse cached ICESat-2 parquet(s); "
                "requesting fresh ATL06-SR data.\n"
            )
    if not loaded_from_reuse:
        icesat.request_atl06sr_multi_processing(
            processing_levels=["all"],
            save_to_parquet=True,
            **atl06sr_time_kwargs,
        )

    icesat.filter_esa_worldcover(filter_out="water")

    # Compute dh (includes 3-sigma outlier filtering by default)
    icesat.atl06sr_to_dem_dh()

    # Resolve the profile track + best/worst segments ONCE so the
    # profile and segment figures (and their aligned variants) all show
    # the same track/segments. When replaying, pin the prior run's
    # choices; otherwise pin this run's auto-selection for self-
    # consistency (#121).
    auto_sel = icesat.get_altimetry_selections("all")
    track_kw = ctx.reuse_track or auto_sel.get("profile_track") or {}
    seg_kw = ctx.reuse_segments or auto_sel.get("segments")
    icesat2_selections = dict(auto_sel)
    if track_kw:
        icesat2_selections["profile_track"] = track_kw
    if seg_kw:
        icesat2_selections["segments"] = seg_kw
    ctx.icesat2_selections = icesat2_selections

    fig_fn = ctx.next_fig_fn()
    icesat.mapview_plot_atl06sr_to_dem(
        key="all",
        save_dir=ctx.plots_directory,
        fig_fn=fig_fn,
        map_crs=ctx.map_crs,
        **ctx_kwargs_altimetry,
    )
    sections.append(
        ReportSection(
            title="ICESat-2 ATL06-SR Map",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.ICESAT2_MAP,
        )
    )

    fig_fn = ctx.next_fig_fn()
    icesat.histogram_by_landcover(
        key="all",
        save_dir=ctx.plots_directory,
        fig_fn=fig_fn,
    )
    sections.append(
        ReportSection(
            title="ICESat-2 ATL06-SR Histogram",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.ICESAT2_HISTOGRAM,
        )
    )

    fig_fn = ctx.next_fig_fn()
    icesat.plot_atl06sr_dem_profile(
        key="all",
        segments=seg_kw,
        save_dir=ctx.plots_directory,
        fig_fn=fig_fn,
        **track_kw,
    )
    sections.append(
        ReportSection(
            title="ICESat-2 ATL06-SR Profile",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.ICESAT2_PROFILE,
        )
    )

    fig_fn = ctx.next_fig_fn()
    icesat.plot_best_worst_segments(
        key="all",
        segments=seg_kw,
        save_dir=ctx.plots_directory,
        fig_fn=fig_fn,
        **track_kw,
    )
    sections.append(
        ReportSection(
            title="ICESat-2 ATL06-SR Agreement Segments",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.ICESAT2_SEGMENTS,
        )
    )

    # ---- pc_align + ICESat-2 alignment report (Earth only) ----
    if cfg.pc_align:
        align_result = icesat.align_and_evaluate(
            processing_level="all",
            improvement_threshold_pct=5.0,
        )
        stats_row = _stats_row_from_result(align_result)

        align_title = "DEM Alignment with ICESat-2"
        alignment_description = captions.EARTH_ALIGNMENT_DESCRIPTION

        if align_result.status == "insufficient_points":
            sections.append(
                AlignmentReportPage(
                    title=align_title,
                    parameters=align_result.parameters_used,
                    description=alignment_description,
                    status_message=align_result.message,
                )
            )
        elif align_result.status == "no_improvement":
            sections.append(
                AlignmentReportPage(
                    title=align_title,
                    parameters=align_result.parameters_used,
                    stats_row=stats_row,
                    description=alignment_description,
                    status_message=align_result.message,
                )
            )
        elif align_result.status == "success":
            # Page A: alignment parameters + stats + description
            # (the histogram gets its own full page below so the
            # figure isn't squeezed into the remaining space)
            sections.append(
                AlignmentReportPage(
                    title=align_title,
                    parameters=align_result.parameters_used,
                    stats_row=stats_row,
                    description=alignment_description,
                    status_message=align_result.message,
                )
            )

            # Page B: pre/post landcover histogram
            fig_fn = ctx.next_fig_fn()
            icesat.histogram_by_landcover(
                key="all",
                plot_aligned=True,
                save_dir=ctx.plots_directory,
                fig_fn=fig_fn,
            )
            sections.append(
                ReportSection(
                    title="ICESat-2 ATL06-SR Histogram (Aligned DEM)",
                    image_path=ctx.fig_path(fig_fn),
                    caption=captions.ICESAT2_HISTOGRAM_ALIGNED,
                )
            )

            # Page C: profile with aligned DEM
            fig_fn = ctx.next_fig_fn()
            icesat.plot_atl06sr_dem_profile(
                key="all",
                segments=seg_kw,
                plot_aligned=True,
                save_dir=ctx.plots_directory,
                fig_fn=fig_fn,
                **track_kw,
            )
            sections.append(
                ReportSection(
                    title="ICESat-2 ATL06-SR Profile (Aligned DEM)",
                    image_path=ctx.fig_path(fig_fn),
                    caption=captions.ICESAT2_PROFILE_ALIGNED,
                )
            )

            # Page D: best/worst segments with aligned DEM
            fig_fn = ctx.next_fig_fn()
            icesat.plot_best_worst_segments(
                key="all",
                segments=seg_kw,
                plot_aligned=True,
                save_dir=ctx.plots_directory,
                fig_fn=fig_fn,
                **track_kw,
            )
            sections.append(
                ReportSection(
                    title="ICESat-2 ATL06-SR Agreement Segments (Aligned DEM)",
                    image_path=ctx.fig_path(fig_fn),
                    caption=captions.ICESAT2_SEGMENTS_ALIGNED,
                )
            )

    return sections


def _build_altimetry_planetary(ctx: ReportContext, body: str):
    cfg = ctx.config
    asp_dem = ctx.asp_dem
    sections: List[object] = []
    instrument = BODIES[body].altimetry_instrument

    if not cfg.altimetry_csv:
        print(
            f"\n{'='*60}\n"
            f"Planetary altimetry requires a pre-downloaded data file.\n\n"
            f"To obtain {instrument} data for this DEM:\n"
            f"  1. Run: request_planetary_altimetry --dem {asp_dem} --email <your_email>\n"
            f"  2. Wait for the email with a download link\n"
            f"  3. Download and unzip the result\n"
            f"  4. Re-run asp_plot with: --altimetry_csv <path_to_pts_csv.csv>\n"
            f"\nSkipping {instrument} altimetry plots.\n"
            f"{'='*60}\n"
        )
        return sections

    alt = Altimetry(directory=cfg.directory, dem_fn=asp_dem)
    alt.load_planetary_csv(cfg.altimetry_csv)
    alt.planetary_to_dem_dh()

    fig_fn = ctx.next_fig_fn()
    alt.mapview_plot_planetary_to_dem(
        save_dir=ctx.plots_directory,
        fig_fn=fig_fn,
    )
    sections.append(
        ReportSection(
            title=f"{instrument} Altimetry Map",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.planetary_altimetry_map(instrument),
        )
    )

    fig_fn = ctx.next_fig_fn()
    alt.histogram_planetary_to_dem(
        save_dir=ctx.plots_directory,
        fig_fn=fig_fn,
    )
    sections.append(
        ReportSection(
            title=f"{instrument} Altimetry Histogram",
            image_path=ctx.fig_path(fig_fn),
            caption=captions.planetary_altimetry_histogram(instrument),
        )
    )

    # ---- pc_align + planetary alignment report (Moon/Mars) ----
    if cfg.pc_align:
        align_result = alt.align_and_evaluate_planetary()
        stats_row = _stats_row_from_result(align_result)

        align_title = f"DEM Alignment with {instrument}"
        alignment_description = captions.planetary_alignment_description(instrument)

        if align_result.status == "insufficient_points":
            sections.append(
                AlignmentReportPage(
                    title=align_title,
                    parameters=align_result.parameters_used,
                    description=alignment_description,
                    status_message=align_result.message,
                )
            )
        elif align_result.status == "no_improvement":
            sections.append(
                AlignmentReportPage(
                    title=align_title,
                    parameters=align_result.parameters_used,
                    stats_row=stats_row,
                    description=alignment_description,
                    status_message=align_result.message,
                )
            )
        elif align_result.status == "success":
            sections.append(
                AlignmentReportPage(
                    title=align_title,
                    parameters=align_result.parameters_used,
                    stats_row=stats_row,
                    description=alignment_description,
                    status_message=align_result.message,
                )
            )

            fig_fn = ctx.next_fig_fn()
            alt.mapview_plot_planetary_to_dem(
                plot_aligned=True,
                save_dir=ctx.plots_directory,
                fig_fn=fig_fn,
            )
            sections.append(
                ReportSection(
                    title=f"{instrument} Altimetry Map (Aligned DEM)",
                    image_path=ctx.fig_path(fig_fn),
                    caption=captions.planetary_altimetry_map_aligned(instrument),
                )
            )

            fig_fn = ctx.next_fig_fn()
            alt.histogram_planetary_to_dem(
                plot_aligned=True,
                save_dir=ctx.plots_directory,
                fig_fn=fig_fn,
            )
            sections.append(
                ReportSection(
                    title=f"{instrument} Altimetry Histogram (Aligned DEM)",
                    image_path=ctx.fig_path(fig_fn),
                    caption=captions.planetary_altimetry_histogram_aligned(instrument),
                )
            )

    return sections


# ---------------------------------------------------------------------------
# The declarative registry: section order is data, not source-line position.
# ---------------------------------------------------------------------------

REPORT_SECTIONS: List[ReportSpec] = [
    ReportSpec("input_scenes", lambda ctx: True, _build_input_scenes),
    ReportSpec(
        "stereo_geometry",
        lambda ctx: bool(ctx.config.plot_geometry),
        _build_stereo_geometry,
    ),
    ReportSpec("match_points", lambda ctx: True, _build_match_points),
    ReportSpec(
        "bundle_adjust",
        lambda ctx: bool(ctx.config.bundle_adjust_directory),
        _build_bundle_adjust,
    ),
    ReportSpec("disparity", lambda ctx: True, _build_disparity),
    ReportSpec("dem_results", lambda ctx: True, _build_dem_results),
    ReportSpec("detailed_hillshade", lambda ctx: True, _build_detailed_hillshade),
    ReportSpec("altimetry", lambda ctx: ctx.plot_altimetry, _build_altimetry),
]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_plot_altimetry(config: ReportConfig) -> bool:
    """Resolve the deprecated --plot_icesat alias against --plot_altimetry."""
    plot_altimetry = config.plot_altimetry
    if config.plot_icesat is not None:
        import warnings

        warnings.warn(
            "--plot_icesat is deprecated. Use --plot_altimetry instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        plot_icesat = config.plot_icesat
        # Convert Click string 'True'/'False' to bool
        if isinstance(plot_icesat, str):
            plot_icesat = plot_icesat.lower() not in ("false", "0", "no")
        plot_altimetry = plot_icesat
    return plot_altimetry


def _setup_context(config: ReportConfig) -> ReportContext:
    """Resolve paths, projection, DEM metadata, and shared plotters."""
    directory = os.path.expanduser(config.directory)
    config.directory = directory
    if config.reference_dem:
        config.reference_dem = os.path.expanduser(config.reference_dem)
    if config.altimetry_csv:
        config.altimetry_csv = os.path.expanduser(config.altimetry_csv)

    print(f"\nProcessing ASP files in {directory}\n")

    plots_directory = os.path.join(directory, "tmp_asp_report_plots/")
    os.makedirs(plots_directory, exist_ok=True)

    # ---- Load reusable figure selections (issue #121) ----
    # Replays a prior run's ICESat-2 points, profile track, best/worst segments,
    # and detailed-hillshade clip boxes so re-processing runs are comparable.
    reuse_clip_windows = None
    reuse_clip_windows_crs = None
    reuse_icesat = {}
    if config.reuse_selections:
        reuse_selections = os.path.expanduser(config.reuse_selections)
        print(f"\nReusing figure selections from: {reuse_selections}\n")
        prior = read_selections_yaml(reuse_selections)
        if prior.detailed_hillshade:
            clips = prior.detailed_hillshade.get("clips") or []
            reuse_clip_windows = [c.get("bbox") for c in clips] or None
            reuse_clip_windows_crs = prior.detailed_hillshade.get("dem_crs")
        reuse_icesat = prior.icesat2 or {}
    reuse_track = reuse_icesat.get("profile_track") or {}
    reuse_segments = reuse_icesat.get("segments")
    reuse_parquet = reuse_icesat.get("parquet_cache")

    report_title = config.report_title
    if report_title is None:
        report_title = os.path.split(directory.rstrip("/\\"))[-1]

    report_filename = config.report_filename
    if report_filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_filename = f"asp_plot_report_{report_title}_{timestamp}.pdf"

    # If report_filename is an absolute or relative path, use it directly;
    # otherwise save alongside the processed DEM in the stereo directory.
    if os.path.dirname(report_filename):
        report_pdf_path = os.path.expanduser(report_filename)
    else:
        report_pdf_path = os.path.join(
            directory, os.path.join(config.stereo_directory, report_filename)
        )

    # Initialize StereoPlotter early (needed for DEM info and multiple plot types)
    stereo_plotter = StereoPlotter(
        directory,
        config.stereo_directory,
        reference_dem=config.reference_dem,
        dem_fn=config.dem_filename,
        dem_gsd=config.dem_gsd,
    )
    asp_dem = stereo_plotter.dem_fn

    # Set map CRS from output DEM and collect DEM metadata
    map_crs = config.map_crs
    report_metadata = None
    if map_crs is None:
        if asp_dem and os.path.exists(asp_dem):
            try:
                dem_raster = Raster(asp_dem)
                epsg_code = dem_raster.get_epsg_code()
                map_crs = f"EPSG:{epsg_code}"
                print(f"\nUsing map projection from DEM: {map_crs}\n")
            except Exception as e:
                print(
                    f"\nError getting projection from DEM: {e}. Using default projection EPSG:4326. If you want a different projection, use the --map_crs flag.\n"
                )
                map_crs = "EPSG:4326"

    # Collect DEM metadata for the report title page
    if asp_dem and os.path.exists(asp_dem):
        try:
            dem_raster = Raster(asp_dem)
            dem_data = dem_raster.read_array()
            total_pixels = dem_data.size
            nodata_pixels = dem_data.mask.sum() if hasattr(dem_data.mask, "sum") else 0
            nodata_pct = (nodata_pixels / total_pixels * 100) if total_pixels else 0.0
            valid = dem_data.compressed()
            elev_range = (
                (float(valid.min()), float(valid.max())) if valid.size else (0, 0)
            )
            acq_extra_dirs = [os.path.join(directory, config.stereo_directory)]
            if config.bundle_adjust_directory:
                acq_extra_dirs.append(
                    os.path.join(directory, config.bundle_adjust_directory)
                )
            acquisition_dates = get_acquisition_dates(
                directory, extra_dirs=acq_extra_dirs
            )
            report_metadata = ReportMetadata(
                dem_dimensions=(dem_raster.ds.width, dem_raster.ds.height),
                dem_gsd_m=dem_raster.get_gsd(),
                dem_crs=map_crs or "",
                dem_nodata_percent=nodata_pct,
                dem_elevation_range=elev_range,
                dem_filename=os.path.basename(asp_dem),
                reference_dem=config.reference_dem or "",
                acquisition_dates=acquisition_dates,
            )
        except Exception as e:
            print(f"\nCould not collect DEM metadata: {e}\n")

    # TODO: Centralize this in plotting utils, should not need ctx import in the CLI wrapper
    if config.add_basemap:
        ctx_kwargs = {
            "crs": map_crs,
            "source": ctx.providers.Esri.WorldImagery,
            "attribution_size": 0,
            "alpha": 0.5,
        }
    else:
        ctx_kwargs = {}

    return ReportContext(
        config=config,
        plots_directory=plots_directory,
        report_pdf_path=report_pdf_path,
        report_title=report_title,
        map_crs=map_crs,
        ctx_kwargs=ctx_kwargs,
        stereo_plotter=stereo_plotter,
        asp_dem=asp_dem,
        plot_altimetry=_resolve_plot_altimetry(config),
        report_metadata=report_metadata,
        reuse_clip_windows=reuse_clip_windows,
        reuse_clip_windows_crs=reuse_clip_windows_crs,
        reuse_track=reuse_track,
        reuse_segments=reuse_segments,
        reuse_parquet=reuse_parquet,
    )


def _write_figure_selections(ctx: ReportContext) -> None:
    """Write the figure-selections sidecar (issue #121)."""
    cfg = ctx.config
    try:
        from asp_plot import __version__ as asp_plot_version
    except Exception:
        asp_plot_version = None

    hillshade_block = None
    hillshade_clips = getattr(ctx.stereo_plotter, "detailed_hillshade_clips", None)
    if hillshade_clips:
        dem_crs_str = None
        try:
            dem_crs_str = str(Raster(ctx.asp_dem).ds.crs)
        except Exception:
            pass
        hillshade_block = {
            "subset_km": float(cfg.subset_km),
            "intersection_error_percentiles": [16, 50, 84],
            "dem_crs": dem_crs_str,
            "clips": hillshade_clips,
        }

    selections = FigureSelections(
        asp_plot_version=asp_plot_version,
        dem_filename=ctx.asp_dem,
        map_crs=ctx.map_crs,
        detailed_hillshade=hillshade_block,
        icesat2=ctx.icesat2_selections,
    )
    selections_path = (
        os.path.splitext(ctx.report_pdf_path)[0] + "_figure_selections.yml"
    )
    try:
        write_selections_yaml(selections_path, selections)
        print(f"\nFigure selections saved to {selections_path}\n")
    except Exception as e:
        print(f"\nCould not write figure selections: {e}\n")


def run_report(config: ReportConfig) -> str:
    """Generate the ASP processing report described by ``config``.

    Builds the runtime context, iterates the declarative
    :data:`REPORT_SECTIONS` registry to assemble the report body, compiles the
    PDF, writes the figure-selections sidecar, and cleans up. Returns the path
    to the written PDF. Importable and callable without any Click context.
    """
    ctx = _setup_context(config)

    sections: List[object] = []
    for spec in REPORT_SECTIONS:
        if spec.enabled(ctx):
            sections.extend(spec.build(ctx))

    # Compile report
    processing_parameters = ProcessingParameters(
        processing_directory=config.directory,
        bundle_adjust_directory=config.bundle_adjust_directory,
        stereo_directory=config.stereo_directory,
    )
    processing_parameters_dict = processing_parameters.from_log_files()

    compile_report(
        sections,
        processing_parameters_dict,
        ctx.report_pdf_path,
        report_title=ctx.report_title,
        report_metadata=ctx.report_metadata,
        report_command=config.report_command,
    )

    _write_figure_selections(ctx)

    shutil.rmtree(ctx.plots_directory)

    print(f"\n\nReport saved to {ctx.report_pdf_path}\n\n")
    return ctx.report_pdf_path
