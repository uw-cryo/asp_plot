# AGENTS.md

This file provides guidance to AI coding agents (Claude Code and others) when working with code in this repository. It is the git-tracked onboarding reference for the package's architecture; keep it in sync with the code.

## Project Overview

`asp_plot` (current version: 1.18.0) is a Python package for visualizing output from the NASA Ames Stereo Pipeline (ASP). It processes stereo satellite imagery results (both terrestrial and planetary), generates diagnostic plots, and creates comprehensive PDF reports similar to those from commercial SfM software like Agisoft Metashape. Requires Python >= 3.11. Published on PyPI and conda-forge.

The package supports:
- Stereo DEM processing visualization
- Bundle adjustment analysis
- CSM camera model comparisons (for tools like bundle_adjust and jitter_solve)
- Altimetry comparisons: ICESat-2 ATL06-SR (Earth), LOLA (Moon), MOLA (Mars)
- Stereo geometry visualization from satellite XML metadata (Earth-based sensors)
- Non-terrestrial/planetary processing (tested with LRO NAC, Mars MRO CTX/HiRISE, Mars MGS MOC, and ASTER)
- Gallery plotting of many DEMs as a grid of thumbnails sharing one color scale

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment (installs package with dev+docs extras)
conda env create -f environment.yml
conda activate asp_plot

# Install pre-commit hooks (REQUIRED for development)
pre-commit install
```

### Testing
```bash
# Run all tests
pytest

# Run tests with output visible (useful for debugging)
pytest -s

# Run a specific test file
pytest tests/test_stereo.py

# Run a specific test function
pytest tests/test_stereo.py::test_function_name
```

### Linting and Formatting
Pre-commit hooks automatically run on commit. To run manually:
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Format code with black
black asp_plot/ tests/

# Lint with flake8
flake8 asp_plot/ tests/

# Sort imports with isort (profile: black)
isort --profile black asp_plot/ tests/
```

Flake8 configuration is in `.flake8` (extends ignore: E203, E701). Pre-commit further extends ignore to: E501, E722, E203, E207.

### Rebuilding the Package
When making changes to CLI tools or entry points:
```bash
pip install -e ".[dev]"
```

### Building Docs Locally
```bash
# One-time: copy notebooks, reports, and figures for local preview
mkdir -p docs/examples/notebooks && cp notebooks/**/*.ipynb docs/examples/notebooks/
mkdir -p docs/_static/reports && cp reports/*.pdf docs/_static/reports/
mkdir -p docs/_extra/examples/figures && cp notebooks/figures/* docs/_extra/examples/figures/

# Build once
sphinx-build docs docs/_build/html

# Or live-reloading preview
sphinx-autobuild docs docs/_build/html --open-browser
```

## Code Architecture

### Core Module Structure

**`__init__.py`** exports `__version__` via `importlib.metadata` (reads from `pyproject.toml` at install time, falls back to `"unknown"`).

The package is organized by functionality, with each module focused on a specific aspect of ASP output visualization. A structural rewrite (issue #122, v1.17.0) split several monoliths into single-concern modules — most of the registry/source/adapter modules below date from that work.

**`bodies.py`** - Single source of truth for per-body planetary facts (issue #126)
- `Body`: frozen dataclass bundling one body's facts — `name`, `altimetry_instrument` (ICESat-2 / LOLA / MOLA), `iau_sphere_radius_m`, `datum` (pc_align `D_MARS`/`D_MOON`), `geocentric_proj` (PROJ string for `apply_dem_translation`), `geographic_crs_wkt`, and ellipsoid fallback (`semi_major_axis_m`, `inverse_flattening`)
- `BODIES`: registry dict keyed `"earth"`/`"moon"`/`"mars"`. Replaced the ad-hoc `{"moon": ..., "mars": ...}` literals that were duplicated across ~40 sites in `alignment.py`, the altimetry sources, `utils.py`, and the CLI
- `body_for_dem(dem_fn, body=None)`: resolves a `Body` from an explicit name or by auto-detecting via `detect_planetary_body()`
- Module constants `MARS_IAU_SPHERE_RADIUS` / `MOON_IAU_SPHERE_RADIUS` (in `planetary_source.py`) are sourced from this registry

**`utils.py`** - Shared utilities for the entire package
- `Raster`: Wrapper class for raster operations using GDAL/rasterio/rioxarray
  - Supports both georeferenced and non-georeferenced (raw) imagery
  - `transform` property returns `None` for non-georeferenced data (identity transform)
  - `get_epsg_code()`: Returns the EPSG code of the raster's CRS. Falls back to the horizontal (2D) component's EPSG via `pyproj` `CRS.to_2d()` when there is no exact match (compound / 3D-promoted CRSs like `"EPSG:32610+EPSG:4979"`, e.g. COP30 DEMs with ellipsoid heights asserted)
  - `get_utm_epsg_code()`: Estimates UTM EPSG code from raster center coordinates
  - `get_gsd()`: Returns ground sample distance in meters
  - `get_bounds()`: Returns bounds using `self.ds.bounds` (rasterio), with optional lat/lon transform
  - `_mask_nodata()`: Private helper that handles both undeclared nodata (inferred from pixel data) and invalid values (inf/nan) — needed because rasterio's `masked=True` alone doesn't catch undeclared nodata in ASP outputs
  - `_load_and_diff_rasters_da()`: Private static method returning xarray DataArray for raster differencing; used internally by both `load_and_diff_rasters()` (converts to numpy) and `compute_difference()` (uses `.rio.to_raster()` for saving)
- `Plotter`: Base class for all plotting classes with common matplotlib setup. The scaffold added in #129 centralizes the boilerplate that was copy-pasted across plot methods: `save(fig, save_dir, fig_fn, ...)` (the `tight_layout()` + conditional-save tail), `plot_missing(ax, message)` (the "required files are missing" placeholder), and `plot_array(..., copyright=True)` which draws the Vantor/WorldView overlay from `self.is_vantor` so the check no longer threads through every call site
- `ColorBar`: Handles colorbar creation and formatting. `get_norm(clim=...)` takes an explicit clim so callers (e.g. `plot_geodataframe`) keep clim local instead of mutating `self.cb.clim`
- File utilities: `glob_file()`, `save_figure()`, `show_existing_figure()`
- Coordinate utilities: `get_utm_epsg()` for determining UTM EPSG from lon/lat, `get_planetary_bounds()` for DEM bounds in planetocentric 0-360 lon/lat
- Planetary body detection: `detect_planetary_body(dem_fn)` returns `"earth"`, `"moon"`, or `"mars"` by inspecting CRS WKT DATUM/ELLIPSOID fields
- Subprocess utilities: `run_subprocess_command()`
- Vantor/copyright utilities (an **attribution** concern — named for the rights-holder — kept deliberately distinct from sensor/reader **identity**, the WorldView-named abstraction in `sensors.py`; #137):
  - `detect_vantor_satellite(directory)`: True when an XML camera file's `SATID` matches `VANTOR_SATID_PREFIXES` — any DigitalGlobe→Maxar→Vantor-owned satellite, i.e. the WorldView family (`WV*`, incl. Legion `WVLG`), GeoEye (`GE*`), QuickBird (`QB*`), IKONOS (`IK*`) — not just WorldView. Gates the `© Vantor` overlay via `Plotter.is_vantor`
  - `add_copyright_overlay(ax)`: Adds "© Vantor {year}" text overlay to bottom-right of matplotlib axes
- Scene metadata: `get_acquisition_dates(directory, extra_dirs=None)` reads `FIRSTLINETIME` from WorldView/Maxar XMLs and parses the capture timestamp from `AST_L1A_...` file/directory names. Returns a sorted, deduplicated list of `"YYYY-MM-DD HH:MM:SS UTC"` strings; empty if nothing is found. Used by the CLI to populate `ReportMetadata.acquisition_dates`.

**`selections.py`** - Reproducible "figure selections" for run-to-run comparison (issue #121)
- `FigureSelections` dataclass mirroring the YAML sidecar schema (`detailed_hillshade` clips + `icesat2` track/segments/parquet/request); `to_dict()`/`from_dict()`
- `write_selections_yaml()` / `read_selections_yaml()` round-trip (uses `pyyaml`)
- `pixel_window_to_bbox()` / `bbox_to_pixel_offset()`: convert a detailed-hillshade clip between a DEM-CRS map bbox and a top-left pixel offset. `bbox_to_pixel_offset` uses `rowcol(..., op=round)` (round-to-nearest, **not** floor) so a replayed clip doesn't drift by a pixel from floating-point error
- `reproject_bbox(bbox, src_crs, dst_crs)`: reprojects a clip bbox between CRSs (via `rasterio.warp.transform_bounds`) so clip reuse works across stereo variants in different projections (e.g. MOC non-mapproj Stereographic vs mapproj Sinusoidal). `plot_detailed_hillshade(clip_windows_crs=...)` passes the manifest's `dem_crs`; reprojection is a no-op when CRSs match
- Deliberately imports nothing from `report.py` / `fpdf`, so it is safe to use from notebooks
- The `asp_plot` CLI always writes `<report_stem>_figure_selections.yml` next to the report; `--reuse_selections PATH` replays a prior run's choices

**`report.py`** - PDF report generation using fpdf2
- `ReportSection`: Dataclass representing a report figure (title, image path, caption)
- `AlignmentReportPage`: Dataclass for the pc_align + ICESat-2 alignment workflow (title, parameters dict, 1-row stats dict, description paragraph, status message, optional figure + caption). Rendered alongside `ReportSection` by `compile_report()`. Body text blocks are left-aligned (not justified) to avoid word-spacing gaps on long lines. Long pc_align column names (`north_shift`, `east_shift`, `down_shift`, `translation_magnitude`) are displayed as `N_shift`, `E_shift`, `D_shift`, `|T|` via `_ALIGNMENT_STATS_DISPLAY_LABELS` so the 10-column horizontal stats row fits in the page width.
- `ReportMetadata`: Dataclass for DEM metadata displayed on the title page (dimensions, GSD, CRS, nodata %, elevation range, DEM filename, reference DEM, acquisition dates). The "Acquisition Date(s)" row is added to the summary table only when `acquisition_dates` is non-empty.
- `ASPReportPDF`: FPDF subclass with custom header/footer and page numbers
- `compile_report()`: Assembles title page, Processing Parameters (page 2), figure sections with captions, and any trailing alignment pages into a PDF. Accepts optional `report_command` string to record the CLI invocation. Figures are automatically scaled to fit page dimensions, preventing overflow/cutoff. `sections` is a mixed list of `ReportSection | AlignmentReportPage`; dispatch is by `isinstance` check.
- `_add_processing_parameters_page()`: helper that renders the runtime summary table plus the bundle_adjust / stereo / point2dem / report commands on page 2 (moved from the trailing page as of v1.13.0). Also renders the reconstructed `mapproject` command(s) (from the optional `mapproject` list key; see `mapproject.py`) with a "reconstructed from output metadata" note.
- `_fmt_sig()`: formats a number compactly — 2 decimals for |x| < 10, 1 decimal for 10 ≤ |x| < 100, 0 decimals above, "n/a" for non-finite. Used for alignment stats.
- Title page displays: processing date, ASP version (from logs), asp_plot version (from package metadata)
- Page order: title + DEM summary → Processing Parameters → diagnostic figures → (if `--pc_align` ran) alignment report page + aligned-DEM figures.
- `report.py` is fed declaratively by `report_pipeline.py`; it was **not** rewritten in #128.

**`report_pipeline.py`** - Declarative report pipeline behind the `asp_plot` CLI (issue #128)
- `ReportConfig`: dataclass packing the ~18 CLI options into one Click-free object (field names/defaults mirror the options one-for-one, guarded by a test)
- `run_report(config)`: importable/callable from notebooks and tests with no Click context; returns the written PDF path
- A declarative section registry (`REPORT_SECTIONS`) of `ReportSpec`s replaces the old inline plot-and-append wall: each spec pairs an `enabled(ctx)` predicate with a `build(ctx)` function returning the sections to append. `--plot_geometry` / `--plot_altimetry` / `--pc_align` gating are predicates; figure numbering comes from a per-run counter on the shared `ReportContext`, so section order and numbering are data, not source-line position. The alignment "Page B/C/D" follow-ups are one spec emitting several sections
- Section builders: `_build_input_scenes`, `_build_stereo_geometry`, `_build_match_points`, `_build_bundle_adjust`, `_build_disparity`, `_build_dem_results`, `_build_detailed_hillshade`, `_build_altimetry` (→ `_build_altimetry_earth` / `_build_altimetry_planetary`)

**`report_captions.py`** - Caption + description text for the report sections (issue #128)
- All section captions and the Earth/planetary alignment descriptions as module constants / small builder functions, moved out of the CLI as a data module (verified byte-identical to the pre-refactor strings via AST comparison)

**`asp_log.py`** - Versioned adapter for parsing ASP log files (issue #132)
- Replaces the hardcoded string surgery that used to live inline in `processing_parameters.py`. `AspLogFormat` describes how to read one ASP log layout (version banner, timestamps, command line, reference DEM) via documented patterns; tool invocations are located by executable basename (against `ASP_TOOL_NAMES`) rather than arbitrary substring matches
- `AspLog`: a single parsed log file, exposing `asp_version`, `command` / `canonical_command` / `tool`, `timestamps` / `first_timestamp` / `last_timestamp`, `reference_dem`
- `select_format()` picks the adapter by version banner; unrecognized banners fall back to `DEFAULT_FORMAT` with a warning; unparseable fields return `None` and log (instead of being swallowed by a bare `except`). `register_format()` is the extension point for future ASP format drift. `STEREO_STEP_ORDER` generalizes stereo step ordering (earliest/latest stage present) instead of hardcoded pprc/tri

**`processing_parameters.py`** - `ProcessingParameters` class
- Delegates all ASP-log parsing to `asp_log.AspLog` (no more inline string surgery; the bare `except:` clauses became directory guards)
- Extracts command lines, run times, and processing parameters; `from_log_files()` returns a dict including the `asp_version` key and a `mapproject` key (list of reconstructed mapproject commands; see `mapproject.py`)
- Used by report generation to document processing settings

**`mapproject.py`** - Reconstruct `mapproject` commands from output GeoTIFF metadata (issue #96)
- ASP's `mapproject` writes **no log file** (unlike `bundle_adjust`/`stereo`/`point2dem`), so `asp_log.py` has nothing to parse for the mapprojection step. Instead of requiring a new ASP `--log` flag, the command is reconstructed **from the output data alone**: ASP stamps `INPUT_IMAGE_FILE` / `CAMERA_FILE` / `DEM_FILE` / `CAMERA_MODEL_TYPE` / `BUNDLE_ADJUST_PREFIX` into each mapprojected GeoTIFF header, and the raster's own CRS / resolution / bounds give `--t_srs` / `--tr` / `--t_projwin`
- `reconstruct_mapproject_command(raster_path)`: returns the `mapproject ...` string, or `None` if the ASP mapproject tag signature (`REQUIRED_TAGS` = `INPUT_IMAGE_FILE` + `CAMERA_FILE` + `DEM_FILE`, all read back during reconstruction) is absent. Reuses the `utils.Raster` wrapper (free `NotGeoreferencedWarning` suppression + `get_epsg_code()` with the compound-CRS 2D fallback + `get_gsd()`/bounds) rather than re-opening with raw rasterio. `--t_srs` is `EPSG:XXXX` when an EPSG code exists (incl. the 2D fallback), else the quoted PROJ string (custom planetary/local frames, e.g. jitter stereographic); a malformed CRS returns `None` and logs instead of crashing the report. `_format_coord` renders coordinates to 12 significant figures, positionally — clean for large UTM northings *and* full-precision for degree-scale geographic `--tr`, with no scientific notation or float-repr noise. The reconstruction is faithful but **not byte-for-byte re-runnable** (session is the resolved `-t`, an input `--mpp` shows as the resolved `--tr`, output reads the actual filename) — the report flags this with a one-line note
- `find_mapproject_commands(directories, stereo_command=None)`: scans dirs (processing root, BA dir, stereo dir) for all `*.tif`/`*.tiff` and keeps those carrying the tag signature — **identity is decided by the file's own metadata, never by filename**, so there is no naming-convention dependency (reading a GeoTIFF header is cheap; the `NotGeoreferencedWarning` from raw non-georef inputs is silenced). Dedupes by the reconstructed command string (identical scene reached via two dirs collapses; distinct left/right both show). When `stereo_command` is given, a discovered output is kept only if its filename appears in that command — this scopes the result to the run being reported, so a non-mapprojected run sharing a parent dir with mapprojected scenes (the `stereo/` + `stereo_no_mapproj/` layout) does **not** spuriously list a mapproject step. It's a whole-token basename membership test (the output basename must equal one of the stereo command's argument basenames — not a raw substring, so `run.tif` can't match `prun.tif`), not positional parsing. `ProcessingParameters.get_mapproject_commands(stereo_command)` passes the parsed stereo command; `report.py` renders the results under "Mapproject Command(s)" on the Processing Parameters page (via the module-level `_render_command_block` helper, shared with the bundle_adjust/stereo/point2dem commands)

**`sensors.py`** - Sensor-specific scene metadata readers (issue #25)
- `SensorMetadata` ABC defining the reader interface (`detect` + `detect_files` + `get_scene_dicts`) and the sensor-agnostic scene-dict schema, mirroring the `bodies.py` registry pattern so adding ASTER/HiRISE/etc. is a new subclass with no change to the geometry code
- `WorldViewMetadata(SensorMetadata)`: the WorldView/Maxar XML logic (file discovery, `dg_mosaic` tiling, per-scene extraction, ephemeris/attitude/footprint) moved verbatim out of the parser. Named after the satellite family rather than the (twice-renamed: DigitalGlobe → Maxar → Vantor) company; the same format also covers GeoEye-1/QuickBird/IKONOS. Constructible from either a `directory=` (recursive discovery) or an explicit `image_list=` (already-resolved XML paths); non-camera files (`README.XML`, ortho) are filtered out either way
- `SENSORS` registry + two detection entry points: `sensor_for_directory()` (scans a directory) and `sensor_for_inputs()` (accepts a mixed list of files/dirs/globs). The latter routes through `resolve_xml_inputs()`, which expands directories (recursively), globs, and plain paths into a deduped, sorted XML list — this is what lets `stereo_geom *.XML` / explicit files / a delivery dir all work without a fixed structure

**`stereopair_metadata_parser.py`** - `StereopairMetadataParser` class
- Now a **sensor-agnostic orchestrator** (issue #25): detects a reader via `sensor_for_directory()` (from `directory=`) or `sensor_for_inputs()` (from a mixed `inputs=` list of files/dirs/globs), delegates scene discovery/extraction to it, and keeps only the pair-level geometry
- Computes stereo geometry metrics: convergence angle, base-to-height ratio, bisector elevation, asymmetry
- Provides spatial utilities: `get_pair_utm_epsg()`, `get_intersection_bounds()`, `get_scene_bounds()`
- `get_catid_dicts()` returns dictionaries per catalog ID with ephemeris, attitude, and geometry data (sourced from the sensor reader)
- **N-scene aware** (issue #73): `get_pair_dict()` is the exact-two-scene entry point (raises otherwise); `get_pair_dicts()` returns one pair dict per N-choose-2 combination so >2 scenes can be assessed pairwise. `get_pair_map_projection()` falls back to the footprint union when a pair does not overlap, and `get_pair_intersection()` tolerates a `None` intersection (non-overlapping pairs report `intersection_area = None`)

**`stereo.py`** - `StereoPlotter` class (inherits from `Plotter`)
- Visualizes ASP stereo processing results
- Plots DEMs, hillshades, disparity maps, match points
- Creates difference maps with reference DEMs
- Supports both map-projected and raw (non-georeferenced) imagery
- Detects map-projection status via `Raster.transform` check
- For non-mapprojected scenes: match points are overlaid on images using alignment transform matrices (`run-align-{L,R}.txt` loaded via `np.loadtxt`), and disparity plots use pixel-unit scalebar instead of GSD-based
- Detects Vantor (WorldView) satellite via `is_vantor` attribute; adds copyright overlay to optical imagery in `plot_match_points()` and `plot_detailed_hillshade()`
- `plot_detailed_hillshade()` auto-selects three subset clips from intersection-error variance (low/medium/high) via `_auto_hillshade_clip_offsets()`. Accepts `clip_windows` (DEM-CRS bboxes) + `clip_windows_crs` to pin/replay clips for run-to-run comparison (issue #121); records the boxes it drew on `self.detailed_hillshade_clips`. Out-of-bounds pinned boxes warn and fall back to auto.
- Key methods: `plot_dem_results()`, `plot_disparity()`, `plot_match_points()`, `plot_detailed_hillshade()`

**`bundle_adjust.py`** - Four main classes
- `ReadBundleAdjustFiles`: Reads bundle adjustment CSV outputs (residual pointmaps)
  - `get_initial_final_residuals_gdfs()`: Returns initial and final residual GeoDataFrames
  - `get_initial_final_geodiff_gdfs()`: Returns geodiff comparison GeoDataFrames (requires `--mapproj-dem` flag in bundle_adjust)
  - `get_mapproj_residuals_gdf()`: Returns map-projected residual GeoDataFrame
- `PlotBundleAdjustFiles` (inherits from `Plotter`): Visualizes bundle adjustment residuals before/after optimization
- Plots include map views of residuals, histograms, and geodiff comparisons
- **Camera before/after position/orientation (issues #95, #43)** — visualizes where each camera *moved*, not just the ground residuals. Unlike `csm_camera.py` (which needs the user to pass original + optimized cameras), this is **self-contained on a bundle_adjust output folder** — the pre-BA original cameras are not co-located there.
  - `ReadBundleAdjustCameras`: discovery is driven by the `*.adjust` files (rigid ECEF translation `T` + rotation quaternion). Each camera's absolute center is anchored at the **center image line** (sub-satellite point at mid-acquisition, more meaningful than the trajectory mean when the ephemeris is padded beyond the image; falls back to the mean if the timing can't be computed). It comes from its `*.adjusted_state.json` (WorldView/CSM + jitter runs; via `getTimeAtLine` + ephemeris interp) or, for DigitalGlobe runs that write only `.adjust` deltas, from the original camera `.xml` ephemeris (`<EPHEMLIST>` interpolated at `FIRSTLINETIME + (NUMROWS/2)/AVGLINERATE`, via `get_xml_tag`; the XMLs are auto-found in the BA dir + its parent, or via `original_cameras_directory` / CLI `--original_cameras_directory`). The two paths agree to <1 m on the same scenes. `get_camera_optimization_gdf(map_crs, original_cameras_directory)` returns one row per camera with the translation decomposed into local ENU (`t_east/t_north/t_up`, `t_horizontal`), the adjustment `adj_roll/adj_pitch/adj_yaw`, and `horizontal_offset_m/vertical_offset_m` (from `camera_offsets.txt` when present — authoritative, folds in the rotation lever-arm — else derived from `T`; flagged by `offsets_from_asp`). The offsets are associated to cameras **positionally** by zipping `camera_offsets.txt` with `camera_list.txt` (both written per input image, in the same order) and keying by the camera-file basename — no `run-`/`_corr` filename-string guessing (`_camera_label` strips only deterministic file extensions for display). Cameras whose center cannot be located (no state file and no matching XML) are warned and skipped.
  - Per ASP's `.adjust` convention (a world point projects the same in the original camera as `R*(P−C)+C+T` in the adjusted, `C` = camera center for pixel (0,0)), `T` is the exact bulk camera-center shift at the anchor pixel; the rotation only adds a lever-arm shift for other lines. `.adjust` naming is matched as both `<base>.adjust` and `<base>.adjusted_state.adjust` (ASTER jitter).
  - `PlotBundleAdjustCameras` (inherits from `Plotter`): `plot_center_offset_bars()` (per-camera horizontal + vertical center change; numeric x-labels in the summary), `plot_orientation_cartoons()` (a per-camera satellite/sensor-frustum cartoon with fixed-length roll/pitch/yaw arrows labeled with the actual degrees changed — the number carries the magnitude, so ~1e-4° noise isn't visually exaggerated; `_draw_satellite` draws one cell), and `summary_plot()` stacking bars over the cartoon grid. The earlier map-view/orientation quivers were dropped as misleading (sub-meter shifts can't be drawn to scale on a ~400 km map; a camera-index x-axis gave quiver direction no meaning). CLI: `bundle_adjust_cameras`.

**CSM camera model comparison** — split into three layers by issue #131 (was one 1541-LOC `csm_camera.py`):

**`csm_io.py`** - Camera-model I/O readers mirrored from ASP's `orbit_plot.py`
- Function-based, with a provenance header noting they are synced from upstream ASP. `read_csm_cam()`, `read_tsai_cam()`, `read_frame_csm_cam()`, `read_linescan_pos_rot()`, `read_angles()`, `toCsmPixel()`, `isLinescan()`, `roll_pitch_yaw()`, `estim_satellite_orientation()`, etc.
- `stereo_geometry.py` imports `estim_satellite_orientation` from here directly

**`csm_analysis.py`** - asp_plot-specific analysis built on `csm_io`
- `get_orbit_plot_gdf()`: turns an original/optimized camera pair into the position- and orientation-difference GeoDataFrame consumed by the plotting layer
- `reproject_ecef()`, `poly_fit()`

**`csm_camera.py`** - Plotting layer for CSM camera optimization / jitter results
- Compares original vs optimized CSM camera models (from bundle_adjust/jitter_solve); analyzes position/orientation differences along the trajectory. Currently supports linescan cameras (e.g., WorldView)
- `csm_camera_summary_plot()`: the near-verbatim cam1/cam2 halves (~210 duplicated lines) collapsed into a single `_plot_camera()` called once per camera (`_apply_frame_xaxis()` for the shared linescan tick logic). Figure output verified unchanged by golden line-content characterization tests
- Re-exports the moved `csm_io`/`csm_analysis` symbols for backward compatibility with notebooks and downstream imports

**Altimetry** — the 3800-line `Altimetry` god-class was split (issues #130, #140) into a thin coordinator plus a source/plotter/base layer. The public API and the `asp_plot.altimetry` re-exports are preserved by delegation, so `report_pipeline.py`, the CLI, and notebooks are unchanged.

**`altimetry.py`** - `Altimetry` coordinator + `AlignmentResult`
- Composes three collaborators — `self.icesat2` (`Icesat2Source`), `self.planetary` (a `PlanetarySource` subclass), and `self.plotter` (`AltimetryPlotter`) — each holding a back-reference to the coordinator, which owns the cross-cutting `directory` / `dem_fn` / `aligned_dem_fn`. The planetary source class is chosen **once, at construction**, from the DEM's body: `{"moon": LolaSource, "mars": MolaSource}.get(detect_planetary_body(dem_fn), PlanetarySource)`
- The full public/notebook API (`request_atl06sr_multi_processing`, `load_planetary_csv`, `atl06sr_to_dem_dh`, `planetary_to_dem_dh`, the `plot_*`/`histogram*`/`mapview_*` methods, `to_csv_for_pc_align*`, the `atl06sr_processing_levels*` / `planetary_points` properties, …) is preserved as **delegating wrappers** over the collaborators; plotting receives already-prepared dataframes (the coordinator computes the dh columns and resolves the track before delegating, so render no longer triggers heavy I/O mid-figure)
- Keeps the `pc_align` orchestration and the shared keep/discard decision: `alignment_report()`, `align_and_evaluate()` (Earth) / `align_and_evaluate_planetary()` (planetary) and the helpers `_improvement_pct` / `_evaluate_improvement` / `_success_result` (the shared decision factored out in #127). Both `align_and_evaluate*` return an `AlignmentResult` dataclass with `status ∈ {"insufficient_points", "no_improvement", "success"}`; the aligned DEM is removed on the non-success branches so its on-disk existence is a truthy "alignment worth using" signal. On success the Earth path re-calls `atl06sr_to_dem_dh(n_sigma=None)` to populate `icesat_minus_aligned_dem` without re-filtering. `Altimetry`, `AlignmentResult`, `ICESAT2_MISSION_START` stay importable from `asp_plot.altimetry`; none of this imports `report.py` / `fpdf`, so it is notebook-safe
- New dependency from this subsystem: `pyyaml` (request-metadata YAML); `sliderule>=5.3.0` pinned

**`altimetry_source.py`** - `AltimetrySource` base (issue #140)
- Shared machinery lifted out of the ICESat-2 and planetary sources so neither re-implements it: `_interp_dem_at_points(dem, points)` (bilinear DEM sampling; returns sampled values **and** the points reprojected into the DEM/working CRS), `_open_dem(dem_fn)`, `_std_outlier_mask(dh, n_sigma)` (n-σ std-from-mean mask, returns `None` on empty/all-NaN or zero spread to signal "do not filter"), and `_write_csv_to_directory(df, filename)` (roots pc_align CSVs at `self.alt.directory`)

**`icesat2_source.py`** - `Icesat2Source(AltimetrySource)` (Earth / ICESat-2)
- Requests/processes ICESat-2 ATL06-SR via the SlideRule x-series API (`sliderule_api.run("atl03x")`) with lazy init (`_ensure_sliderule()` — connects only when ICESat-2 methods run). SlideRule logging silenced (`verbose=False`, WARNING, explicit filter on `sliderule.session`)
- `request_atl06sr_multi_processing()` decomposed (#140) into a short request/cache/ingest loop over helpers: `_print_time_filter_summary`, `_build_level_parms`, `_load_cached_atl06sr`, `_params_match_cache`, `_request_atl06sr_level`, `_ingest_atl06sr`. Parquet cache saved to `self.alt.directory` (not CWD); the param comparison strips the `output` key (SlideRule injects a random temp path into `parms` during `run()`) and only the comparison is wrapped in `try/except` so SlideRule API errors propagate
- Server-side time filtering via `_resolve_time_range(...)`: default `"all"` = full mission (2018-10-14 → today UTC midnight); `"buffered"` cascades explicit `t0`/`t1` > `scene_date` ± `time_buffer_days` > XML metadata ± buffer > `"all"`. `t1` truncated to midnight UTC for stable caching
- ESA WorldCover sampling: `sample_esa_worldcover()` / `_sample_worldcover_into_gdf()` read 10m values directly from public AWS S3 COGs (`esa-worldcover.s3.amazonaws.com/v200/2021/map/`) via `rasterio` vsicurl, batched per tile, persisted into the parquet cache (much faster than SlideRule's server-side `samples`). `_worldcover_tile_url(lat, lon)` maps a coord to the 3×3° tile URL (flat layout). Anonymous reads (unsigned AWS session) so SSO configs don't crash
- `filter_outliers(n_sigma=3)`: drops dh beyond `n_sigma × std` from the mean (true std, not NMAD); called automatically by `atl06sr_to_dem_dh()`; pass `None` to skip
- `atl06sr_to_dem_dh()`: opens the DEM via `self._open_dem`, samples via `self._interp_dem_at_points`, computes the dh column. Track/segment selection: `_select_best_track()` (most valid dh points), `_find_best_worst_segments()` / `_segment_dict()` / `_resolve_best_track()`
- **Run-to-run reuse (issue #121)**: `load_atl06sr_from_parquet()` replays the *exact* prior points (bypassing SlideRule), `_restore_request_metadata_from_parquet()` restores `t0`/`t1` for plot titles, `get_altimetry_selections()` returns request params / parquet paths / track / segments. Segments pinned by **absolute `x_atc`** (not km-from-track-start) so they survive a track-start shift when 3σ filtering a different DEM drops a different first point
- `_extract_scalar()` handles array-valued cells from the x-series API / parquet round-trip; module constant `ICESAT2_MISSION_START`

**`planetary_source.py`** - `PlanetarySource(AltimetrySource)` + `LolaSource` / `MolaSource` (issue #140)
- Two-step workflow: (1) submit async query via `request_planetary_altimetry` CLI → email with download link, (2) load the CSV. The body subclass is selected by the coordinator at construction, so loading no longer re-detects the body per call
- `PlanetarySource` (body-agnostic base): the dh + export half — `_load_planetary_csv_common()`, `_build_planetary_gdf()`, `_find_csv_column()`, `_sample_dem_at_planetary_points()` (uses `self._open_dem` / `self._interp_dem_at_points`), `planetary_to_dem_dh()` (uses `self._std_outlier_mask`; when `aligned_dem_fn` is set also samples the aligned DEM so pre/post plots share one sample), `to_csv_for_pc_align_planetary()` (writes `lon, lat, radius_m`). Its `load_planetary_csv()` raises and redirects to ICESat-2 (the Earth case)
- `LolaSource` (Moon, `instrument = "LOLA"`): prefers `Pt_Radius` (LOLA RDR Point-per-Row CSV, in **km** — auto-detected by magnitude < 10 000, converted to m), falls back to `Topography` (simple-topo CSV, meters). Moon ≈ spherical (~1.4 km variation) so the fallback is fine
- `MolaSource` (Mars, `instrument = "MOLA"`): reads `PLANET_RAD` from the ODE GDS `*_pts_csv.csv`, `height = PLANET_RAD - 3_396_190`. The `*_topo_csv.csv` (TOPOGRAPHY only) is **rejected** with an explanatory error — TOPOGRAPHY is referenced to the **oblate** MOLA areoid while ASP DEMs use the **spherical** IAU 2000 datum, a latitude-dependent offset up to ~10 km that pc_align can't remove
- Both loaders store `height` (above the IAU sphere; plot labels + dh stats) and `radius_m` (absolute planetary radius; pc_align) on `planetary_points`
- Module-level `gds_query_async()` + `GDS_BASE_URL` (`https://oderest.rsl.wustl.edu/livegds`), re-exported from `asp_plot.altimetry` so the CLI import is unchanged; constants `MARS_IAU_SPHERE_RADIUS` / `MOON_IAU_SPHERE_RADIUS` (sourced from `BODIES`). `LolaSource` / `MolaSource` / `PlanetarySource` are also re-exported from `asp_plot.altimetry`

**`altimetry_plots.py`** - `AltimetryPlotter` (issue #130)
- All figure rendering, operating on already-prepared dataframes. `histogram_by_landcover()`, `plot_atl06sr_dem_profile()` (2×2: stacked elevation + dh on the left, DEM hillshade map spanning both rows on the right), `plot_best_worst_segments()` (scoring `3·|median(dh)| + NMAD(dh)`, weighting bias 3× over dispersion; requires >75% DEM coverage per segment), `mapview_plot_planetary_to_dem()` / `histogram_planetary_to_dem()`, plus the over-long profile/segment methods decomposed into panel helpers (`_profile_elevation_panel`, `_profile_dh_panel`, `_draw_segment_panel`, `_plot_hillshade_map`, `_add_segment_spans`, `_build_landcover_stats_text`)
- `plot_aligned=True` overlays pre/post-alignment data on shared color/bin scales; dh maps + histograms use symmetric ±|3σ min/max| clim centered on 0 (labeled `[±3σ]`); all data is still plotted and used for Median/NMAD stats. Module constant `WORLDCOVER_NAMES`; the shared `nmad` helper now lives in `utils.py`
- Requires internet for data requests (SlideRule + AWS S3 WorldCover COGs)

**`alignment.py`** - `Alignment` class
- Performs DEM alignment using ASP's `pc_align` tool
- `pc_align_dem_to_atl06sr()`: ICESat-2 path, csv-format `1:lon 2:lat 3:height_above_datum`
- `pc_align_dem_to_planetary_csv(planetary_csv, body, ...)`: MOLA/LOLA path. Uses csv-format `1:lon 2:lat 3:radius_m` and `--datum D_MARS`/`D_MOON` (per ASAP-Stereo's CTX cookbook). Default `max_displacement=500` m
- Both public methods keep their signatures/validation/errors and delegate to a shared `_run_pc_align(csv, csv_format, max_displacement, datum=...)` (#127); generated argv is byte-identical to before, with `--datum` emitted only on the planetary path
- `pc_align_report()`: Extracts begin/end percentiles + N-E-D translation from the pc_align log
- `apply_dem_translation()`: Applies pc_align's Cartesian translation to the DEM (geotransform shift + scalar add to pixel values, no resampling). Picks the right body-centered geocentric source CRS via the module-level `_GEOCENTRIC_PROJ` dict — Earth uses EPSG:4978, Mars/Moon use PROJ strings (`+proj=geocent +R=...`) because PROJ refuses to convert across celestial bodies
- Used by `Altimetry` class for DEM-to-altimetry alignment on Earth, Mars, and Moon

**`stereo_geometry.py`** - `StereoGeometryPlotter` class
- **Composes** a `StereopairMetadataParser` via `self.parser` rather than subclassing it (issue #25); imports `estim_satellite_orientation` from `csm_io` directly
- Visualizes stereo acquisition geometry from XML metadata
- `dg_geom_plot()`: Creates skyplot (satellite viewing angles) and map view (footprints)
- `satellite_position_orientation_plot()`: Creates 3x2 figure showing position covariance, roll/pitch/yaw orientation, and attitude covariance for each scene

**`scenes.py`** - `ScenePlotter` class (inherits from `Plotter`)
- Plots individual input scenes (satellite/spacecraft images)
- Works with any sensor (terrestrial or planetary)
- Automatically detects and indicates whether scenes are map-projected or raw
- Displays filenames rather than sensor-specific metadata
- Detects Vantor (WorldView) satellite via `is_vantor` attribute; adds copyright overlay to scene images in `plot_scenes()`
- Key method: `plot_scenes()` (formerly `plot_orthos()` prior to v1.2.0)
- Used in comprehensive reports to show source imagery

**`gallery.py`** - `GalleryPlotter` class (inherits from `Plotter`)
- Lays out a *stack* of DEMs as a grid of thumbnails sharing one global percentile color stretch and a single colorbar (for QA'ing multi-date / multi-pair ASP output at a glance). Replaces the legacy `original_code/gallery.py` and its `pygeotools`/`imview` deps; reuses `Raster` (downsampled reads), `ColorBar.find_common_clim`, and `save_figure`
- Renders DEMs with the package's standard convention: gray hillshade underlay + semi-transparent `viridis` DEM, `"Elevation (m HAE)"` colorbar (matches `stereo.py` `_plot_dem_with_hillshade`). Hillshade underlay on by default
- `from_directory(directory, pattern="*-DEM.tif")` resolves a directory + glob into the raster list; globs with `recursive=True` so `**` descends into subdirectories (the per-pair ASP layout). Also accepts an explicit file list via `__init__`
- Layout sizes each panel to the rasters' median aspect ratio and places panels with absolute positioning, so 1 to N rasters (incl. non-square) pack tightly without internal or trailing-cell whitespace
- Per-panel titles use the full filename, auto-shrunk to fit the panel by *measuring* rendered text width with an Agg renderer (`_fit_titles()`); falls back to a character-count heuristic (`_fit_title_fontsize()`) if measurement fails
- Output detail is read at ~`GALLERY_TARGET_PX` (1200) px/panel and save dpi is matched to it for crisp zoom, then dpi is capped to a pixel budget so the PNG stays under `max_filesize_mb` (default 10) regardless of raster count
- Key method: `plot_gallery()`; static helper `_grid_shape(n, aspect)` picks the most square-in-display grid
- Not wired into the main `asp_plot` PDF report (standalone class + `gallery` CLI)

### CLI Tools

All CLI tools are in `asp_plot/cli/` and use Click for argument parsing:

**`asp_plot.py`** - Main CLI tool (`asp_plot` command)
- Generates comprehensive PDF reports of ASP processing
- A **thin Click wrapper** (issue #128): parses the ~18 options into a `ReportConfig` and calls `report_pipeline.run_report(config)`. All orchestration lives in `report_pipeline.py` (see the module structure section); the CLI file itself is now just option definitions + the `ReportConfig` build
- Accepts directories for stereo and bundle_adjust outputs
- Options for reference DEMs, ICESat-2 comparisons, basemaps
- Report section order: Input Scenes → Stereo Geometry → Match Points → Bundle Adjust panels (Log/Linear Residuals, Map-Projected Residuals, Geodiff) → Disparity → DEM Results → Detailed Hillshade → Altimetry panels. Disparity follows Bundle Adjust (not Hillshade) and DEM Results precedes Hillshade, so the title page → inputs → per-step diagnostics → final DEM products narrative reads top-down.
- Title-page `ReportMetadata` is populated here; `get_acquisition_dates()` is called with the main directory plus stereo/BA subdirs as `extra_dirs` so the Acquisition Date(s) row appears on both raw-input layouts (WV XMLs at top level) and layouts where XMLs live one level deep.
- `--directory`: Root ASP processing directory (default: `./`)
- `--stereo_directory`: Stereo output subdirectory (default: `stereo`)
- `--bundle_adjust_directory`: Optional BA directory
- `--dem_filename`: Custom DEM filename (default: auto-detect `*-DEM.tif`)
- `--dem_gsd`: Custom DEM ground sample distance
- `--map_crs`: Projection as `EPSG:XXXX` (default: auto-detect from DEM, fallback `EPSG:4326`)
- `--reference_dem`: Reference DEM path (auto-detected from logs if not supplied)
- `--add_basemap`: Add Esri WorldImagery basemaps (default: True, requires internet)
- `--plot_altimetry`: Plot altimetry comparisons (default: True). Auto-detects planetary body from DEM CRS: Earth → ICESat-2 (requires internet), Moon → LOLA, Mars → MOLA. For planetary DEMs, requires `--altimetry_csv`. Replaces the deprecated `--plot_icesat` flag.
- `--plot_icesat`: Deprecated alias for `--plot_altimetry`. Prints deprecation warning if used.
- `--altimetry_csv`: Path to a LOLA/MOLA CSV from the ODE GDS API. **Mars: must be the `*_pts_csv.csv` (not `*_topo_csv.csv`) — the loader requires the `PLANET_RAD` column to avoid the oblate-areoid offset.** Moon accepts either the `*_topo_simple_csv.csv` (results=u) or the `*_pts_csv.csv` (results=p). Obtained via the `request_planetary_altimetry` CLI tool.
- `--pc_align`: If True (default) and `--plot_altimetry` is True, runs `pc_align` against the reference altimetry (ICESat-2 for Earth, MOLA for Mars, LOLA for Moon) after the existing altimetry plots and appends an alignment report. **Earth success path**: adds four pages (alignment report page, pre/post landcover histogram, aligned profile, aligned best/worst segments). **Planetary success path**: adds three pages (alignment report page, pre/post mapview, pre/post histogram). `insufficient_points` and `no_improvement` outcomes emit a single alignment report page on either branch. Disabled automatically when `--plot_altimetry` / `--plot_icesat` is False.
- `--plot_geometry`: Plot stereo geometry (default: True; disable for planetary missions)
- `--subset_km`: Hillshade subset size in km (default: 1.0)
- `--atl06sr_time_range`: Time range for ICESat-2 ATL06-SR requests. `"all"` (default) for full mission, `"auto"` for scene metadata ±1 year, `"START,END"` for a custom range, or a single date (buffered by ±1 year).
- `--reuse_selections`: Path to a `*_figure_selections.yml` from a prior run. Replays that run's ICESat-2 points (parquet), profile track, best/worst segments, and detailed-hillshade clips so re-processing runs (e.g. mapproj vs non-mapproj) are directly comparable (issue #121). Every run always writes `<report_stem>_figure_selections.yml` next to the report (the `regenerate_reports.sh` paired variants use this to reuse each other). Generated sidecars are gitignored (they hardcode absolute local paths); a sanitized example is in `docs/cli/asp_plot.md`.
- `--report_filename`: PDF report filename or path. A bare filename saves in the stereo directory; a path (e.g. `reports/report.pdf`) is used as-is
- `--report_title`: Custom report title (default: directory name)

**`csm_camera_plot.py`** - CSM camera comparison tool (`csm_camera_plot` command)
- Wrapper for `csm_camera.py` functions
- Compares original and optimized CSM camera models
- Visualizes position/angle differences and camera footprints

**`bundle_adjust_cameras.py`** - Self-contained camera before/after position tool (`bundle_adjust_cameras` command)
- Wrapper for `ReadBundleAdjustCameras` + `PlotBundleAdjustCameras` in `bundle_adjust.py`
- Takes a single `--directory` = the bundle_adjust output folder (split internally into the reader's root+subdir); no original cameras needed. `--save_dir` defaults to that same folder
- Renders the three-panel `summary_plot()` (position-change quiver, center-displacement bars, orientation-change quiver)

**`request_planetary_altimetry.py`** - Planetary altimetry data request tool (`request_planetary_altimetry` command)
- Submits async LOLA (Moon) or MOLA (Mars) queries to the ODE GDS REST API
- Auto-detects planetary body from DEM CRS via `detect_planetary_body()`
- `--dem`: Path to ASP DEM (required)
- `--email`: Email for notification when query finishes (required)
- `--channels`: LOLA detector channels (Moon only, default `tffff` = channel 1 only)
- Saves request metadata as `altimetry_request_info.yml` alongside the DEM
- Workflow: submit query → receive email → download/unzip → pass `*_pts_csv.csv` (Mars) or `*_topo_simple_csv.csv`/`*_pts_csv.csv` (Moon) to `asp_plot --altimetry_csv`

**`stereo_geom.py`** - Stereo geometry visualization tool (`stereo_geom` command)
- Wrapper for `StereoGeometryPlotter`
- Creates skyplot and map view from XML camera files
- Supports multiple XMLs with automatic mosaicking
- Accepts positional `INPUTS` (any mix of XML files, directories, and globs — e.g. `stereo_geom *.XML`); falls back to `--directory` when no positional inputs are given. Both paths funnel into the same plotter (`inputs=` vs `directory=`)
- **N-scene output** (issue #73): two scenes → one `<name>_stereo_geom.png` (unchanged). More than two → one color-coded overview figure (`_overview.png`) plus one figure per pair (`_<catidA>_<catidB>.png`, or `_pairN.png` if a CATID is missing), each with full pairwise stats. `dg_geom_plot()` dispatches on scene count and returns the list of saved filenames; `StereoGeometryPlotter` adds `_render_pair`/`_render_overview`. Map views keep the full (autoscaled) extent so all satellite ephemeris tracks are visible (off-nadir scenes have tracks tens to hundreds of km from the footprints), with tracks drawn on top of the semi-transparent footprints; `_add_basemap_safe` falls back to a coarse zoom so the wide-extent basemap fetch doesn't fail on contextily's negative auto-zoom

**`gallery.py`** - DEM gallery tool (`gallery` command)
- Wrapper for `GalleryPlotter`; lays out many DEMs as a grid sharing one color scale
- `--directory` + `--pattern` (supports recursive `**` for subdirectories) or an explicit list of `FILES` (files take precedence)
- `--hillshade/--no-hillshade`, `--cmap`, `--downsample`, `--max_filesize_mb`, `--title`, `--output_directory/--output_filename`
- Saves `<dirname>_gallery.png` into the input directory by default

## Documentation Website

Documentation is hosted at **https://asp-plot.readthedocs.io** and built with Sphinx + MyST Markdown + sphinx-book-theme. ReadTheDocs builds automatically on push to `main`.

### Architecture

- **Stack**: Sphinx, myst-nb (Markdown + notebook rendering), sphinx-autoapi (API docs via static analysis, no package import needed), sphinx-design (cards/grids), sphinx-book-theme (collapsible sidebar)
- **Config**: `docs/conf.py` — Sphinx configuration; `.readthedocs.yaml` — RTD build config
- **RTD build deps**: `docs/requirements.txt` — installed instead of the full package (avoids GDAL on RTD)
- **Version fallback**: `conf.py` uses `try/except` for `importlib.metadata.version()` since the package isn't installed on RTD

### Content Structure

```
docs/
  conf.py                     # Sphinx configuration
  index.md                    # Landing page with sphinx-design cards
  installation.md             # conda/pip/source install
  cli/                        # CLI tool docs (asp_plot, stereo_geom, csm_camera_plot, request_planetary_altimetry, gallery)
  examples/
    index.md                  # Notebook gallery with cards by sensor
    reports.md                # PDF reports embedded as iframes
    notebooks/                # .gitignored — copied from notebooks/ during build
  figures/                    # Committed doc figures (e.g. example_gallery.png); referenced from .md via ../figures/
  contributing.md             # Dev setup, testing, release process
  changelog.md                # Includes top-level CHANGELOG.md via MyST include
  _static/reports/            # .gitignored — copied from reports/ during build
  _extra/examples/figures/    # .gitignored — copied from notebooks/figures/ during build
  _templates/footer.html      # RTD ethical ads placement in footer
  requirements.txt            # Docs-only pip dependencies for RTD
```

### Key Design Decisions

- **Notebooks and reports stay at top level** (`notebooks/`, `reports/`). RTD's `pre_build` jobs in `.readthedocs.yaml` copy them into the docs tree. The `docs/examples/notebooks/`, `docs/_static/reports/`, and `docs/_extra/` directories are `.gitignore`d.
- **sphinx-autoapi** generates API reference from static analysis — no GDAL/rasterio needed on RTD. This is why RTD installs only `docs/requirements.txt` instead of the full package.
- **`html_extra_path`** is used to serve `notebooks/figures/` at the correct relative path for notebook `<img src>` references.
- **`docs/` is excluded from sdist** in `pyproject.toml` so docs never ship in the PyPI/conda package.
- **Changelog** uses `{include} ../CHANGELOG.md` so there's one source of truth.
- **Selective notebook exclusion**: `.readthedocs.yaml`'s `pre_build` copies every `notebooks/**/*.ipynb` into `docs/examples/notebooks/`, but specific notebooks can be dropped from the build by listing them in `exclude_patterns` in `docs/conf.py` (e.g., `worldview_utqiagvik_stereo.ipynb`). Inter-notebook links in the WorldView examples use fully-qualified `https://asp-plot.readthedocs.io/en/latest/...` URLs so they resolve both on RTD and in raw notebook previews. Report-link convention: each notebook's "Full Report Generation" section ends with a `#### See the resulting [report](https://asp-plot.readthedocs.io/en/latest/_static/reports/<filename>.pdf).` line that links directly to the PDF served from `_static/reports/`. The `<filename>` must match what the cell's `!asp_plot --report_filename` writes (so the URL on RTD actually resolves).

### Dependencies

Docs dependencies are in `pyproject.toml` under `[project.optional-dependencies] docs` and mirrored in `docs/requirements.txt` (for RTD). The `environment.yml` installs with `pip install -e ".[dev,docs]"`.

## Key Design Patterns

**Inheritance for shared scaffolding, composition for collaborators**: Plotting classes inherit from the `Plotter` base for consistent matplotlib setup and helpers (`save`, `plot_missing`, `plot_array`); the altimetry sources inherit `AltimetrySource` for shared DEM-sampling/outlier/CSV helpers. But cross-concern wiring is *composed*, not inherited (the #122 rewrite moved several god-classes this way): `Altimetry` composes its sources + plotter, `StereoGeometryPlotter` composes a `StereopairMetadataParser`, `StereopairMetadataParser` composes a `SensorMetadata` reader, and `StereoPlotter`/`ScenePlotter` compose a `*Files` discovery object.

**Registries over conditionals**: per-target facts live in one registry rather than scattered `if body == ...` / `if sensor == ...` branches — `BODIES` (`bodies.py`), `SENSORS` (`sensors.py`), `ASP_LOG_FORMATS` (`asp_log.py`), `REPORT_SECTIONS` (`report_pipeline.py`). Adding a 4th body / 3rd sensor / new ASP-log format / reordered report section is a new registry entry, not edits across many files.

**Lazy Loading**: Data is loaded on-demand (e.g., DEMs, hillshades) rather than in `__init__` to avoid unnecessary I/O.

**Internal rioxarray, Public numpy**: rioxarray is used internally where it simplifies implementations (e.g., raster reprojection/alignment in `_load_and_diff_rasters_da()`, saving via `.rio.to_raster()`). The public API always returns `numpy.ma.MaskedArray`. Avoid opening files redundantly — prefer reusing the existing `self.ds` (rasterio dataset) for metadata like bounds and transforms.

**External Tool Integration**: The package wraps ASP command-line tools (pc_align, geodiff, dg_mosaic) via `run_subprocess_command()`.

**Report Generation**: The main `asp_plot` CLI uses `compile_report()` to combine individual plots into a single PDF with metadata tables.

## ASP Tool Dependencies

This package is designed to work with outputs from the NASA Ames Stereo Pipeline. Key ASP tools used:
- `stereo` / `parallel_stereo`: Main stereo processing (generates DEMs, disparity maps, match files)
- `bundle_adjust`: Camera optimization (generates residual pointmaps)
- `pc_align`: Point cloud alignment (used for DEM-to-altimetry alignment)
- `geodiff`: Generates difference statistics between DEMs
- `dg_mosaic`: Mosaics multiple XML files for tiled imagery
- `point2dem`: Converts point clouds to DEMs

## External Data Sources

**ICESat-2 ATL06-SR** (Earth): Fetched via SlideRule x-series API (`sliderule_api.run("atl03x")`, requires internet). The `Altimetry` class handles requests with automatic server-side time filtering, data normalization, land cover filtering, and comparison with ASP DEMs.

**LOLA RDR** (Moon): Queried via `request_planetary_altimetry` CLI (`query=lolardr`). User receives an email with a download link. Either the simple-topography CSV (`results=u`, columns `Pt_Longitude, Pt_Latitude, Topography`) or the Point per Row CSV (`results=p`, includes `Pt_Radius` in **kilometers**) works. The Moon is essentially spherical (~1.4 km equatorial-vs-polar variation), so LOLA topography ≈ height above the IAU 1737.4 km lunar sphere to ~1 m. Loaded via `Altimetry.load_planetary_csv()`.

**MOLA PEDR** (Mars): Queried via `request_planetary_altimetry` CLI (`query=molapedr`). User receives an email with a download link containing both `*_topo_csv.csv` (TOPOGRAPHY only, areoid-referenced) and `*_pts_csv.csv` (full-fidelity record including `PLANET_RAD`). **Use the `*_pts_csv.csv`.** The loader computes `height = PLANET_RAD - 3,396,190` (IAU 2000 Mars sphere) and rejects the topo-only file with an explanatory error. Reason: MOLA TOPOGRAPHY is referenced to the **oblate** MOLA areoid (~20 km equatorial-vs-polar variation) while ASP DEMs use the **spherical** IAU sphere; dh from TOPOGRAPHY carries a latitude-dependent offset of up to ~10 km that pc_align cannot remove (verified at lat 34°N: MOC NA dh dropped from +6000 m to ~+25 m simply by switching to PLANET_RAD).

**ODE GDS REST API**: Base URL `https://oderest.rsl.wustl.edu/livegds`. Queries are submitted via `gds_query_async()` in async mode. The `request_planetary_altimetry` CLI submits the query and the user downloads results via email link. Coordinates use east-positive 0-360 longitude and planetocentric latitude.

**Basemaps**: Uses `contextily` to fetch Esri WorldImagery tiles (requires internet). Can be disabled with `--add_basemap False`. Automatically skipped for planetary (non-Earth) DEMs.

## Testing

Tests are in `tests/` with sample data in `tests/test_data/`. Most modules have corresponding test files (e.g., `test_stereo.py` for `stereo.py`). Test data includes synthetic rasters, XML camera files, bundle adjustment CSVs, ICESat-2 parquet files, pc_align outputs, and jitter correction data. There is also a `test_imports.py` that verifies all modules can be imported.

Example notebooks demonstrating modular usage are organized by sensor type:
- `notebooks/WorldView/` - DigitalGlobe/Maxar WorldView examples (Earth-based). Includes a no-mapprojection variant (`worldview_spacenet_atlanta_stereo_without_mapprojection.ipynb`) that runs `parallel_stereo --alignment-method affineepipolar` on the raw `*_corr.tif` images instead of `*_corr_map.tif`. To keep the comparison ROI matched to the mapprojected example, that notebook documents an inverse-RPC trick: open the original `.tif` (which carries RPC metadata; the `wv_correct` `_corr.tif` does not) with GDAL, build a `gdal.Transformer(["METHOD=RPC", "RPC_HEIGHT=<m>"])`, call `TransformPoint(1, lon, lat, h)` (direction `1` = inverse: ground → pixel) on the four corners of the mapprojected notebook's UTM `t_projwin`, take the bounding box, and pass the result as `--left-image-crop-win`. This reuses the existing `ba/` outputs and writes to `stereo_no_mapproj/` so both runs coexist.
- `notebooks/ASTER/` - ASTER examples with map-projection and jitter correction (Earth-based)
- `notebooks/LRO_NAC/` - Lunar Reconnaissance Orbiter Narrow Angle Camera examples (Lunar)
- `notebooks/Mars_MGS/` - Mars Global Surveyor MOC NA examples (Mars). Single notebook covers both mapprojected (`cam2map4stereo.py` first) and non-mapprojected (`--alignment-method affineepipolar`) variants of the M0100115 / E0201461 pair, mirroring the ASTER mapproj/non-mapproj layout
- `notebooks/Mars_MRO/` - Mars Reconnaissance Orbiter CTX and HiRISE examples (Mars)

## Versioning and Release Process

Follow semantic versioning (MAJOR.MINOR.PATCH). To release:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with version and date
3. Merge to `main`

The rest is automated:
- `release.yml` detects the version bump, creates a GitHub Release + tag, then builds and publishes to PyPI via OIDC trusted publishing (uses `softprops/action-gh-release@v2`)
- conda-forge's `regro-cf-autotick-bot` detects the new PyPI version and opens a feedstock PR automatically

**One-time setup** (already completed):
- PyPI trusted publisher configured at pypi.org for `release.yml` (environment name left blank)
- conda-forge feedstock created via `staged-recipes` PR (reference recipe in `conda-forge-recipe/meta.yaml`)
- All runtime dependencies declared in `pyproject.toml` (`pip install asp-plot` installs all deps)
- Documentation hosted on ReadTheDocs (auto-builds on push to `main`)

## Common File Patterns

ASP output files follow specific naming patterns:
- DEMs: `*-DEM.tif` or `*_dem.tif`
- Disparity: `*-F.tif` (disparity map)
- Match files: `*.match`
- Bundle adjust residuals: `*-initial_residuals_pointmap.csv`, `*-final_residuals_pointmap.csv`
- Log files: `*log-bundle_adjust*.txt`, `*log-stereo*.txt`, `*log-point2dem*.txt`

Use `glob_file()` utility to find files matching these patterns.
