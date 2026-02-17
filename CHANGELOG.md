# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.1] - 2026-02-16

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
