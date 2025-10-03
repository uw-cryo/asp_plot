# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- `load_and_diff_rasters()` now directly uses `rasterio.warp.reproject()` instead of geoutils
- `compute_difference()` now returns `(this_raster - second_raster)` and uses this raster's grid as reference (was previously reversed)
- `compute_difference()` no longer saves by default (use `save=True` to enable)
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
