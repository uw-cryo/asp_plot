# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.14.0] - 2026-04-28

### Added
- **Automatic `pc_align` step in the planetary altimetry block** ([#119](https://github.com/uw-cryo/asp_plot/pull/119)). The existing `--pc_align` CLI flag now also runs against MOLA (Mars) and LOLA (Moon) — previously Earth/ICESat-2 only. Mirrors the Earth pipeline: a single alignment-report page on `insufficient_points` / `no_improvement`, plus a pre/post mapview and pre/post histogram on `success`.
- **`Altimetry.align_and_evaluate_planetary(...)`**: planetary sibling of `align_and_evaluate`. Returns the same `AlignmentResult` dataclass; defaults `max_displacement=500` m (per ASAP-Stereo's CTX cookbook) and `minimum_points=20` (planetary tracks are sparse).
- **`Alignment.pc_align_dem_to_planetary_csv(...)`**: invokes ASP `pc_align` with `--csv-format '1:lon 2:lat 3:radius_m'` and `--datum D_MARS`/`D_MOON` (aligned with the [ASP `next_steps` documentation on MOLA alignment](https://stereopipeline.readthedocs.io/en/latest/next_steps.html)).
- **`Altimetry.to_csv_for_pc_align_planetary()`**: writes `lon, lat, radius_m` from `self.planetary_points` to drive `pc_align`.
- **`plot_aligned` kwarg** on `Altimetry.mapview_plot_planetary_to_dem` and `Altimetry.histogram_planetary_to_dem`: pre/post panels share color/bin scales when an aligned DEM is available.
- **Module-level constants** `MARS_IAU_SPHERE_RADIUS = 3_396_190.0` and `MOON_IAU_SPHERE_RADIUS = 1_737_400.0` so callers can reconstruct ASP-style "height above sphere" without magic numbers.

### Changed
- **MOLA loader switched to `PLANET_RAD`**. `_load_mola_csv()` now reads the absolute planetary radius from the ODE GDS `*_pts_csv.csv` and computes `height = PLANET_RAD - 3,396,190` (IAU 2000 Mars sphere). The `*_topo_csv.csv` (TOPOGRAPHY only) is **rejected** with an explanatory error: TOPOGRAPHY is referenced to the **oblate** MOLA areoid while ASP DEMs use the **spherical** IAU 2000 datum, so dh from TOPOGRAPHY carries a latitude-dependent offset of up to ~10 km that `pc_align` cannot remove. Verified on the MOC NA tutorial scene at lat 34°N: signed median dh dropped from +6,000 m (TOPOGRAPHY path) → +99.74 m (PLANET_RAD path) → +3.13 m (after `pc_align`). Reference: [MOLA PEDR Software Interface Specification (PDS Geosciences)](https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-3-pedr-l1a-v1/mgsl_21xx/document/pedrsis.pdf).
- **LOLA loader prefers `Pt_Radius` (km) when available**. The Point per Row LOLA RDR CSV (`results=p`) carries `Pt_Radius` in **kilometers**; the simple Topography CSV (`results=u`) carries Topography in meters. `_load_lola_csv()` auto-detects km by magnitude (< 10 000) and converts to meters, then writes both `height` (m above the IAU 1737.4 km lunar sphere) and `radius_m` to `self.planetary_points`. The Moon is essentially spherical (~1.4 km equatorial-vs-polar variation), so either CSV gives the same dh to ~1 m. Reference: [ODE GDS REST V2.0 manual](https://oderest.rsl.wustl.edu/GDS_REST_V2.0.pdf).
- **`Alignment.apply_dem_translation()` is body-aware**. Picks a body-centered geocentric "ECEF-equivalent" CRS from a new module-level `_GEOCENTRIC_PROJ` dict — Earth uses `EPSG:4978`; Mars/Moon use PROJ strings (`+proj=geocent +R=...`) because PROJ refuses to convert across celestial bodies. Without this fix, applying a `pc_align` translation to a Mars/Moon DEM raised `RuntimeError: Source and target ellipsoid do not belong to the same celestial body`.
- **`planetary_to_dem_dh()` also samples the aligned DEM** when `self.aligned_dem_fn` is set, populating `aligned_dem_height` and `altimetry_minus_aligned_dem` so pre/post plots share a single sample. Refactored shared interpolation into `_sample_dem_at_planetary_points()`.

### Documentation
- **MOC NA notebook consolidated** into `notebooks/Mars_MGS/mars_mgs_orbital_camera.ipynb` covering both stereo variants of the M0100115 / E0201461 pair (the `mars_mgs_orbital_camera_narrow_angle.ipynb` notebook for a different scene pair was removed). Mirrors the ASTER mapproj/non-mapproj layout. Stereo commands match this repo's WorldView convention (`parallel_stereo --stereo-algorithm asp_mgm --subpixel-mode 9 --processes 2 --threads 4`, `--alignment-method affineepipolar` for non-mapprojected, `--alignment-method none` for mapprojected via `cam2map4stereo.py`). The notebook intro includes a callout explaining the spherical-vs-oblate elevation-range surprise. Reports: `MOC-asp-plot-report.pdf` and `MOC_mapproj-asp-plot-report.pdf`.
- **LRO NAC notebook reprocessed on the full 5000×5000 cubes** in [`LRONAC_example.tar`](https://github.com/NeoGeographyToolkit/StereoPipelineSolvedExamples/releases/download/LRONAC/LRONAC_example.tar) instead of the 900×973 sub-window the ASP "lightning fast" tutorial uses. Resulting DEM is 4720×4510 at 1.04 m GSD (~4.7 km × 4.7 km, vs the old ~1 km × 1 km), 95.88% valid pixels. LOLA query expanded to match: 1539 of 2044 LOLA points overlap the DEM (vs 12 of 19 on the old crop), enough for a meaningful `pc_align` and to bring out spacecraft jitter in the disparity panels.

## [1.13.0] - 2026-04-20

### Added
- **Automatic `pc_align` step in the Earth altimetry block**, gated by a new `--pc_align` CLI flag (default `True`; disabled automatically when `--plot_altimetry` / `--plot_icesat` is `False`). Runs `pc_align` against ICESat-2 ATL06-SR, evaluates whether the aligned DEM is worth keeping, and appends the outcome as one or more report pages:
  - Always: an **alignment report page** with the parameters table, a single-row horizontal stats table (p16/p50/p84 beg/end, north/east/down shifts, translation magnitude, values to 2 sig figs), a description explaining what `pc_align` does and the meaning of every column in the tables above, and a bold status line for the outcome of this run.
  - On success (p50 drops toward 0 by more than `improvement_threshold_pct`, default 5%, **and** `pc_align` actually wrote an aligned DEM): three additional full-page diagnostic figures against the aligned DEM — a pre-/post-alignment landcover histogram, the full profile, and the best/worst 1 km segments.
  - On insufficient ATL06-SR coverage or no meaningful improvement: the aligned DEM on disk is cleaned up so its presence is a truthy signal that the alignment is worth using.
- **`Altimetry.align_and_evaluate(...)`** (new method) returning a plain `AlignmentResult` dataclass (`status ∈ {"insufficient_points", "no_improvement", "success"}`, `alignment_report_df`, `aligned_dem_fn`, `improvement_pct`, `message`, `parameters_used`). Does not import any `fpdf` / report dependencies, so it is safe to call from notebooks.
- **`plot_aligned` kwargs** on `Altimetry.histogram_by_landcover` and `Altimetry.plot_best_worst_segments`:
  - `histogram_by_landcover(plot_aligned=True)` overlays the pre- and post-alignment distributions using shared bin edges and renders two vertically stacked per-landcover stats text boxes whose outline colors match the bar colors (color = legend).
  - `plot_best_worst_segments(plot_aligned=True)` keeps segment selection fixed (based on the unaligned `dh` so segments are comparable), overlays aligned DEM heights on each segment, and appends aligned Median/NMAD to the segment titles.
- **`AlignmentReportPage`** dataclass in `asp_plot.report`: a report-section type that renders a kwargs table + single-row stats table + description + bold status line + optional figure with caption. Body text blocks render left-aligned to avoid justified word-spacing gaps.

### Changed
- **Processing Parameters is now page 2 of the PDF**, immediately after the DEM Summary on the title page, instead of the trailing appendix. Page order is now: title + DEM summary → processing parameters → diagnostic figures → (if any) alignment results.
- **`plot_atl06sr_dem_profile(plot_aligned=True)`**: the lower `dh` panel now plots the post-alignment residuals (`icesat_minus_aligned_dem`) with Med/NMAD recomputed against the aligned DEM, with the legend entry tagged `"(Aligned DEM)"`. The upper elevation panel still overlays both the unaligned and aligned DEM for comparison. `plot_aligned=False` behavior is unchanged.

## [1.12.1] - 2026-04-14

### Changed
- **Report panel order**: Disparity maps now follow the Bundle Adjust panels, and DEM Results precedes Detailed Hillshade (was: Hillshade → DEM Results → Disparity).
- **Input Scenes caption** clarifies that mapprojected scenes are RPC-orthorectified against a reference DEM to roughly pre-align the stereo pair prior to correlation (reducing disparity search range), which addresses confusion for readers coming from non-ASP photogrammetry workflows.
- **Match Points caption** notes these come from `stereo_corr`'s initial interest point matching step (used to set search windows), not dense correlation.

### Added
- **Acquisition Date(s)** row on the DEM Summary title-page table, populated when recoverable from scene metadata.
- `get_acquisition_dates()` helper in `utils.py`: reads `FIRSTLINETIME` from WorldView/Maxar XMLs and parses the capture timestamp from `AST_L1A_...` file/directory names. Deduplicates and sorts; returns an empty list if no date can be found, in which case the summary-table row is omitted.
- Unit tests for `get_acquisition_dates()` covering WorldView XMLs, ASTER filenames (top-level and in subdirectories), dedupe, and sorting of multi-date pairs.

### Fixed
- `--report_filename` accepts absolute and relative paths (not just bare filenames), and `~` in CLI path arguments is expanded ([#113](https://github.com/uw-cryo/asp_plot/pull/113)).

## [1.12.0] - 2026-04-10

### Changed
- **ICESat-2 time filter default changed to `"all"`** (full mission range) instead of auto-detect ±1 year. The `--atl06sr_time_range` CLI option now accepts `"all"` (default), `"auto"` (XML metadata ±`time_buffer_days`), `"START,END"`, or a single date (buffered). Programmatic API: `_resolve_time_range()` and `request_atl06sr_multi_processing()` take a new `time_range` parameter (`"all"` or `"buffered"`) with cascade: `t0`/`t1` > `scene_date` > XML metadata > fall back to `"all"`. `t1` is truncated to midnight UTC for stable parquet caching.
- **ESA WorldCover sampled locally from AWS S3 COGs** via `rasterio` vsicurl instead of through the slow SlideRule `samples` parameter. WorldCover is now sampled inside `request_atl06sr_multi_processing` before the parquet save, so the column is persisted in the cache and doesn't have to be re-sampled on subsequent runs. COP30 also sampled alongside (asset name corrected to `esa-copernicus-30meter`).
- **3σ outlier filter** applied by default in `atl06sr_to_dem_dh` (and `planetary_to_dem_dh`) using the true mean ± 3·standard deviation (not NMAD). Pass `n_sigma=None` to skip. Dh colorbars and histograms use symmetric ±|filtered min/max| (≈ ±3σ) centered on 0 as display limits; all data is still plotted. Displayed stats remain Median / NMAD.
- **Profile plot (`plot_atl06sr_dem_profile`) restructured**: stacked elevation/dh plots on the left (shared x-axis, no vertical gap, grid lines), map view on the right spanning both rows. Figure reshaped to 16×8. Dh points colored gray instead of salmon.
- **Best/worst segments (`plot_best_worst_segments`) simplified** to a 1×2 figure (removed the context map). Scoring formula changed from `|median(dh)| + NMAD(dh)` to `3·|median(dh)| + NMAD(dh)` so a large median bias can't be hidden by a small NMAD. Labels "Better agreement" / "Worse agreement" instead of "Best" / "Worst". CLI report caption documents the formula.
- **Parquet cache** saved next to the ASP processing directory (`self.directory`) instead of the current working directory.
- **SlideRule logging** silenced (`verbose=False`, WARNING level, explicit filter on `sliderule.session`).

### Added
- `filter_outliers()` method: removes dh points beyond `n_sigma` × standard deviation from the mean.
- `sample_esa_worldcover()` method: samples ESA WorldCover 10m values from AWS S3 COGs for manually-loaded data (auto-called inside `request_atl06sr_multi_processing`).
- `plot_best_worst_segments()` method: 1×2 figure showing 1 km segments with better and worse ICESat-2 vs DEM agreement.
- ICESat-2 time filtering documentation section in `docs/cli/asp_plot.md` explaining the three modes.

### Fixed
- **Parquet cache regeneration bug**: SlideRule mutates the `parms` dict by injecting a random temp file path at `output.path` during `run()`, causing the string comparison to fail on every subsequent run. `output` is now stripped from both sides of the comparison and from stored parameters.
- **Parquet cache error swallowing**: the broad `try/except` around the cache comparison also wrapped the SlideRule API call; narrowed it so API errors propagate instead of being silently eaten.
- **Histograms no longer cut data**: replaced `range=` (which excludes data outside the range from the bins) with `ax.set_xlim()`, so all data is plotted and used in stats.
- **Single-date CLI argument** now uses `scene_date` buffering instead of being treated as a start date.
- COP30 SlideRule asset name corrected (`esa-copernicus-30meter`, not `cop30-dem`).

### Dependencies
- Pinned `sliderule>=5.3.0` to pick up temp file handling fixes.

## [1.11.1] - 2026-03-30

### Fixed
- Asymmetry angle calculation: ECEF ground point z-coordinate was incorrectly set to 0 (equatorial plane) instead of using the proper WGS84 ellipsoid position from pyproj, producing wrong values at non-equatorial latitudes

### Changed
- Stereo geometry functions (`get_convergence_angle`, `get_bh_ratio`, `get_bie_angle`, `get_asymmetry_angle`) extracted to module-level in `stereopair_metadata_parser.py` for reuse and testability

### Added
- Unit tests for convergence angle, B/H ratio, BIE, and asymmetry angle calculations, including a regression test for the ECEF z=0 bug

## [1.11.0] - 2026-03-26

### Added
- New `--atl06sr_time_range` CLI option for controlling ICESat-2 ATL06-SR time filtering: use `"all"` for full mission range, or `"START,END"` for a custom date range (e.g. `"2020-01-01,2024-12-31"`)
- Corresponding `t0`/`t1` parameters on `Altimetry.request_atl06sr_multi_processing()` and `Altimetry._resolve_time_range()` for programmatic use
- New WorldView-3 UCSD example notebook (`worldview_spacenet_ucsd_stereo.ipynb`) using publicly available IARPA CORE3D data, with comprehensive stereopair selection analysis
- Example report: `WorldView_UCSD-asp-plot-report.pdf`

### Fixed
- `Alignment.pc_align_report()` and `Alignment.apply_dem_translation()` now return `None` gracefully when pc_align log files are not found, instead of crashing with `TypeError`
- `Altimetry.alignment_report()` handles missing pc_align results with a warning instead of crashing
- `key_for_aligned_dem` parameter in `Altimetry.alignment_report()` now defaults to the `processing_level` value instead of being hardcoded to `"ground"`

## [1.10.0] - 2026-03-21

### Added
- Planetary altimetry validation: LOLA (Moon) and MOLA (Mars) DEM comparison via the ODE Granular Data System (GDS) REST API, analogous to the existing ICESat-2 workflow for Earth DEMs
- New `request_planetary_altimetry` CLI tool to submit async LOLA/MOLA data requests with email notification, saving request metadata to `altimetry_request_info.yml`
- New `--plot_altimetry` flag on the `asp_plot` CLI with automatic body detection (Earth → ICESat-2, Moon → LOLA, Mars → MOLA)
- New `--altimetry_csv` flag to pass a pre-downloaded LOLA/MOLA `*_topo_csv.csv` file for planetary altimetry plots
- `detect_planetary_body()` utility function: detects Earth/Moon/Mars from DEM CRS WKT
- `get_planetary_bounds()` utility function: converts DEM bounds to planetocentric 0-360 lon/lat for GDS queries
- `Altimetry.load_planetary_csv()`: loads LOLA or MOLA CSV with column validation and helpful error messages
- `Altimetry.planetary_to_dem_dh()`: computes altimetry-minus-DEM differences using WKT-based CRS (supports planetary DEMs without EPSG codes)
- `Altimetry.mapview_plot_planetary_to_dem()`: DEM hillshade with dh point overlay
- `Altimetry.histogram_planetary_to_dem()`: dh histogram with n/median/NMAD statistics
- Lazy SlideRule initialization: `Altimetry.__init__` no longer requires an internet connection; SlideRule is initialized on first ICESat-2 method call
- LOLA/MOLA altimetry sections added to LRO NAC, Mars MGS MOC, Mars MGS MOC NA, and Mars MRO HiRISE example notebooks
- Unit tests for body detection, planetary bounds, lazy init, CSV loading/validation, and planetary dh computation

### Changed
- `--plot_icesat` is now a deprecated alias for `--plot_altimetry` (prints deprecation warning if used)
- Basemaps are automatically skipped for non-Earth DEMs
- `pyyaml` added as an explicit dependency

## [1.9.0] - 2026-03-10

### Added
- Match points now overlay on non-mapprojected images using alignment transform matrices (`run-align-{L,R}.txt`), replacing the previous blank-right-panel behavior
- Report command string recorded in PDF report via new `report_command` parameter in `compile_report()`
- Pixel-unit scalebar for non-mapprojected disparity plots (mapprojected scenes continue to use GSD-based scalebar)
- Guard with `FileNotFoundError` when alignment matrix files are missing for non-mapprojected match point overlay
- Warning when `unit="meters"` is passed for non-mapprojected disparity (unsupported, falls back to pixels)
- Test coverage for non-mapprojected stereo code paths (9 new tests with resampled ASTER test data)

### Changed
- Report figures are now fitted to page dimensions, preventing overflow and cutoff for large/wide figures
- Report caption reserve is now dynamically calculated from actual caption length instead of a hardcoded 20mm
- Input Scenes caption updated to explain alignment rotation applied to non-mapprojected imagery
- Match points right subplot title simplified from "Right (scenes shown only if mapprojected)" to "Right"
- `save_figure()` default DPI changed from hardcoded 150 to `None` (uses figure's own creation DPI), fixing pixelated ICESat-2 report figures
- ICESat-2 altimetry figures created at 220 DPI for high-quality PDF embedding
- CLI parameter values are now quoted with `shlex.quote()` for proper reconstruction of commands with spaces
- Cleaned up example notebook report links and removed stale PDF files
- Removed unnecessary `read_align_matrix()` method; alignment matrices are loaded inline via `np.loadtxt()`

### Fixed
- Disparity plot scale for non-mapprojected scenes: GSD-based rescale was producing near-zero values from the identity transform; now skips rescaling and uses pixel-unit scalebar instead
- Match point plot whitespace for non-mapprojected scenes caused by a 1x1 dummy image plotted underneath scatter points
- Pixelated ICESat-2 ATL06-SR figures in PDF reports caused by `save_figure()` overriding figure DPI with 150

## [1.8.0] - 2026-03-03

### Added
- New `_select_best_track()` method to find the RGT/cycle/spot combination with the most valid ATL06-SR points for profile plotting
- New `histogram_by_landcover()` method producing a histogram of ICESat-2 vs DEM differences with per-landcover-class statistics (count, median, NMAD) using ESA WorldCover
- New `plot_atl06sr_dem_profile()` method with a three-row figure: combined elevation + dh profile with dual y-axes, two 1 km zoom segments (best/worst agreement scored by |median(dh)| + NMAD), and DEM hillshade map with track overlay
- Server-side time filtering for SlideRule API requests via new `_resolve_time_range()` method with three-tier cascade: explicit `scene_date` parameter, auto-detect from stereopair XML metadata, or 2-year fallback
- `scene_date` and `time_buffer_days` parameters added to `request_atl06sr_multi_processing()`
- Module-level `ICESAT2_MISSION_START` constant and `WORLDCOVER_NAMES` dictionary for reuse across methods
- Module-level `_nmad()` helper function (Normalized Median Absolute Deviation)
- Time range labels displayed on ICESat-2 plot titles
- Tests for `_select_best_track`, `histogram_by_landcover`, `plot_atl06sr_dem_profile`, and `_resolve_time_range`

### Changed
- Migrated SlideRule API from legacy `icesat2.atl06p()` to x-series `sliderule_api.run("atl03x")` with automatic index and column normalization
- Simplified ICESat-2 report section: single `"all"` processing level with landcover histogram and profile plot, replacing the previous multi-level (all + ground) workflow with temporal filtering and plain histograms
- Report section ordering: bundle adjustment plots now appear after match points and before DEM hillshade
- Profile plot legend now includes axis labels (left/right) for all entries and embeds Med/NMAD statistics in the dh legend item

### Removed
- `--icesat_filter_date` CLI option (time filtering is now automatic via `_resolve_time_range()`)
- Commented-out `plot_atl06sr_dem_profiles()` stub and ATL03 request stub (replaced by implemented methods)
- Duplicated WorldCover classification table from `filter_esa_worldcover()` docstring (now references `WORLDCOVER_NAMES`)

### Fixed
- `TypeError: Cannot subtract tz-naive and tz-aware datetime-like objects` in `predefined_temporal_filter_atl06sr` when scene date is UTC-aware but DataFrame index is tz-naive
- `KeyError: 'translation_magnitude'` in `alignment_report()` when requested processing level has no data (now returns early with a warning)
- `TypeError: unhashable type: 'numpy.ndarray'` in `histogram_by_landcover` caused by parquet round-trip deserializing arrays as Python lists
- `TypeError: 'int' object is not callable` when builtin `len()` was shadowed by the `len=40` parameter inside `request_atl06sr_multi_processing`
- `OverflowError: cannot convert float infinity to integer` in profile segment selection when median point spacing is zero

## [1.7.0] - 2026-02-24

### Added
- ASP version and asp_plot version displayed on report title page
- Copyright overlay ("© Vantor {year}") on WorldView satellite imagery in scene, match point, and detailed hillshade plots
- `detect_vantor_satellite()` utility to identify WorldView imagery from XML SATID tags
- `add_copyright_overlay()` utility for matplotlib axes
- `ProcessingParameters.get_asp_version()` method to extract ASP version from log files
- `Raster._mask_nodata()` private helper to consolidate nodata/invalid value masking
- `Raster._load_and_diff_rasters_da()` private static method returning xarray DataArray for raster differencing

### Changed
- `Raster.get_bounds()` now uses `self.ds.bounds` (rasterio) instead of opening a redundant rioxarray dataset
- `Raster.compute_difference()` uses `rio.to_raster()` for saving when `save=True`, avoiding manual profile construction
- `StereoPlotter.plot_detailed_hillshade()` reuses existing `raster.ds.transform` instead of reopening the DEM file
- Consolidated duplicated nodata masking logic into `Raster._mask_nodata()`
- Updated ASTER and WorldView example notebooks

## [1.6.4] - 2026-02-17

### Changed
- Updated README installation instructions with conda-forge as recommended install method
- Updated README release process documentation for automated pipeline

## [1.6.3] - 2026-02-16

### Fixed
- Added missing runtime dependencies to `pyproject.toml`: `geopandas`, `matplotlib-scalebar`, `sliderule`

## [1.6.2] - 2026-02-16

### Added
- Automated PyPI publishing via OIDC trusted publishing on GitHub Release
- conda-forge reference recipe for staged-recipes submission
- Runtime dependencies declared in `pyproject.toml` (`pip install asp-plot` now installs all deps)

### Changed
- Replaced deprecated `actions/create-release@v1` with `softprops/action-gh-release@v2` in release workflow
- Added missing dependencies to `environment.yml`: `pyproj`, `scipy`, `shapely`, `xarray`

## [1.6.0] - 2026-02-16

### Added
- Structured PDF report generation with title page, section headings, figure captions, DEM metadata summary table, and runtime summary table
- New `report.py` module containing `ReportSection` and `ReportMetadata` dataclasses, `ASPReportPDF` class, and `compile_report()` function
- DEM metadata (dimensions, GSD, CRS, nodata %, elevation range) automatically collected and displayed on the report title page
- Figure captions describing each plot in the generated PDF report
- Page headers (report title) and footers (page numbers) throughout the report
- Tests for report dataclasses and PDF compilation (8 new tests)

### Changed
- Replaced `markdown-pdf` dependency with `fpdf2` (available on conda-forge, enabling conda-only installation)
- Reordered report sections: Input Scenes and Stereo Geometry now appear before DEM results, matching the logical processing flow
- Report generation moved from `utils.py` to dedicated `report.py` module
- PNG images are now embedded directly in the PDF (eliminated intermediate PNG-to-JPEG conversion step)

### Removed
- Dependency on `markdown-pdf` (pip-only package that blocked conda-forge packaging)

## [1.5.0] - 2026-02-13

### Added
- Satellite attitude (ATT) parsing from DigitalGlobe/Maxar XML files: new `getAtt()` and `getAtt_df()` methods on `StereopairMetadataParser`, mirroring the existing ephemeris parsing
- New `satellite_position_orientation_plot()` method on `StereoGeometryPlotter` producing a 3x2 figure showing position covariance, roll/pitch/yaw orientation, and attitude covariance for each scene
- Attitude data (`att_df`) now included in catalog ID dictionaries returned by `get_catid_dicts()`

### Changed
- Ephemeris covariance columns in `getEphem_gdf()` renamed from `x_cov, y_cov, ...` to `cov_11, cov_12, cov_13, cov_22, cov_23, cov_33` for clarity

## [1.4.0] - 2025-12-12

### Added
- New WorldView SpaceNet Atlanta stereo processing example notebook using publicly available data
- New utility function `get_utm_epsg()` for determining UTM EPSG code from longitude/latitude
- New `Raster.get_utm_epsg_code()` method for estimating UTM zone from raster location
- New `StereopairMetadataParser` methods: `get_pair_utm_epsg()`, `get_intersection_bounds()`, `get_scene_bounds()`

### Changed
- `Alignment.get_alignment_report()` now returns North-East-Down shift keys (`north_shift`, `east_shift`, `down_shift`) instead of ECEF Cartesian keys (`x_shift`, `y_shift`, `z_shift`).
- Renamed `worldview_comprehensive.ipynb` to `worldview_utqiagvik_stereo.ipynb`.

### Fixed
- Fixed `geodiff` command in bundle adjustment processing: corrected argument order (DEM must come before CSV) and csv-format syntax (spaces instead of commas between column specs)
- Fixed graceful handling when `--mapproj-dem` flag was not used in bundle_adjust: geodiff plots are now skipped with a warning instead of causing the entire bundle adjustment section to fail
- Fixed relative reference DEM paths read from log files not being resolved to absolute paths, causing "file not found" errors

## [1.3.1] - 2025-11-17

### Added
- Jitter solved ASTER example processing notebook

### Fixed
- Currently the GSD for bundle adjustment calculations is pulled from the metadata for WorldView scenes. ASTER does not contain this metadata, so a fallback value is used (1 m GSD), which effectively renders the bundle adjustment calculations always in pixels. We will eventually want to support an argument or other parser, but this is not important at the moment and instead this approach gracefully allows plotting to continue without erroring out.

## [1.3.0] - 2025-11-14

### Added
- Several new example processing notebooks in `notebooks/`
- A new argument `date` to `Altimetry.predefined_temporal_filter_atl06sr`, which can be used to pass the capture date of the scene for filtering. Previously, the date was read from metadata, but that only works for WorldView right now.
- A new flag to the `asp_plot` CLI: `--icesat_filter_date`, which passes the YYYY-MM-DD formatted date to the icesat filtering method

### Fixed
- Previously, when void pixels were contained in the detailed mapprojected subset images in the detailed hillshade plots, the entire subset plot would appear blank. This is fixed by masking no data values and calculating the color ranges excluding them.
- Similarly, the disparity maps were also improperly showing data void areas. This is fixed by better handling of void areas during the disparity map calculations and plotting.

## [1.2.1] - 2025-10-19

### Added
- New `notebooks/Mars_MOC` example

### Fixed
- Added a regular hillshade fallback to `StereoPlotter.plot_detailed_hillshade()` for the case where `*-IntersectionErr.tif` was not produced and is not available for detailed hillshade plots.

### Internal
- Extracted common hillshade plotting logic in `StereoPlotter` to utility function.

## [1.2.0] - 2025-10-12

### Added
- Support for non-terrestrial (planetary) ASP processing, tested with Lunar Reconnaissance Orbiter (LRO) Narrow Angle Camera (NAC) data
- New `--plot_geometry` CLI flag to optionally skip stereo geometry plots (default: True)
- New `--subset_km` CLI flag to configure hillshade subset size in kilometers (default: 1.0 km)
- Example notebook for LRO NAC processing in `notebooks/LRO_NAC/`
- Detection and handling of non-georeferenced (raw, non-map-projected) raster data

### Changed
- **API change**: `ScenePlotter.plot_orthos()` renamed to `ScenePlotter.plot_scenes()` for sensor-agnostic naming
- `ScenePlotter` no longer depends on `StereopairMetadataParser`, making it compatible with non-Earth sensors
- Scene plots now automatically detect and display whether images are map-projected or raw
- Scene plot titles now show filenames instead of Earth-specific metadata (catalog ID, GSD)
- `Raster.transform` property now returns `None` for non-georeferenced images (identity transform) instead of identity Affine
- Suppressed `NotGeoreferencedWarning` when opening non-georeferenced rasters
- Match points plot clarification text updated: "scenes shown only if mapprojected"

### Removed
- `StereoPlotter.is_mapprojected()` method - replaced with simpler `Raster.transform` check

### Internal
- Simplified map-projection detection logic using `Raster.transform is None` check

## [1.1.1] - 2025-10-10

### Changed
- Moved existing example notebooks into `WorldView` sub-directory, since we plan to introduce other sensors and we'd like to keep things separated in our examples.

### Fixed
- While moving and re-running notebooks, it was noted that `Altimetry.plot_atl06` had a bug when `plot_dem=True`. The `rasterio.plot.show` was improperly imported. This is properly imported now.

## [1.1.0] - 2025-10-03

### Added
- `downsample` parameter to `Raster` class for efficient downsampled reading
- Lazy-loaded `data` property on `Raster` class using `@property` decorator
- `save_raster()` static method for flexible raster saving with reference metadata
- Optional `save` parameter (default `False`) to `compute_difference()` method
- `_calculate_downsampled_shape()` private method for modular downsampling logic
- Comprehensive test suite for `Raster` and `ColorBar` classes (21 new tests in `test_utils.py`)
- Explicit `rioxarray` dependency to `environment.yml` (was previously an implicit dependency via geoutils)

### Changed
- Refactored `Raster` class to remove dependency on `geoutils`
- `load_and_diff_rasters()` now uses `rioxarray` for efficient reprojection and cropping (matching geoutils behavior with simpler implementation)
- `compute_difference()` no longer saves by default (use `save=True` to enable)
- Difference rasters are now cropped to the intersection of both input rasters (matching geoutils behavior)
- Updated `altimetry.py` to use native rasterio plotting instead of geoutils

### Removed
- Dependency on `geoutils` (>=0.1.9)
- Dependency on `xdem` (was unused)

### Internal
- Extracted downsampling logic into reusable private method
- Added properties for `data` and `transform` with lazy loading
- Improved separation of concerns between data loading and file I/O

## [1.0.2] - 2025-08-09

### Fixed
- Small typo csm_camera CLI help text
- Improper passing of map_crs into csm_camera CLI tool fixed
- Sometimes while trimming linescan cameras to only the rows of image capture in the csm_camera utilities, the indices of first and last collection line are reversed. I think this has to do with ascending versus descending orbits, but I didn't investigate deeply. I did add a conditional check to the responsible function to switch the index slicing in this case.

## [1.0.1] - 2025-04-27

### Improved
- Added comprehensive docstrings throughout the codebase for better code documentation

## [1.0.0] - 2025-04-27

This is the first stable release of `asp_plot`. While it was previously available as a pre-1.0 package,
this release marks a commitment to proper versioning and documentation.

### Added
- New `stereo_geom` command-line tool for visualizing stereo geometry
- Added comprehensive docstrings to CLI tools
- Created this CHANGELOG file for better tracking of changes
- Added support for multiple XML files with automatic mosaicking via `dg_mosaic`

### Fixed
- Fixed subprocess handling in `stereopair_metadata_parser.py` for multiple XML file processing

## [0.5.10] - 2024-XX-XX

Combined beta release of `asp_plot` since version 0.0.1, before a proper change log was established.

### Added
- Initial public release of `asp_plot`
- Support for bundle adjust visualization
- Support for stereo visualization
- Report generation capabilities
- CSM camera plot tool
