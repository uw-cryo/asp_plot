# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
