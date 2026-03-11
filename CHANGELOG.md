# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
