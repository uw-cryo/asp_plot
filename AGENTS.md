# AGENTS.md

Guidance for AI coding agents (Claude Code and others) and new contributors. This file holds only what you can't infer from the code: commands, gotchas, external dependencies, and process. For the module-by-module codebase map and design rationale, read [ARCHITECTURE.md](ARCHITECTURE.md) on demand — and keep both files in sync with the code.

## Project Overview

`asp_plot` is a Python package for visualizing output from the NASA Ames Stereo Pipeline (ASP): diagnostic plots and comprehensive PDF reports covering stereo DEM results, bundle adjustment, CSM camera models, stereo acquisition geometry, DEM galleries, and altimetry comparison/alignment (ICESat-2 for Earth, LOLA for Moon, MOLA for Mars). Handles terrestrial and planetary sensors. Requires Python >= 3.11; published on PyPI and conda-forge. The version lives in `pyproject.toml` (exposed as `asp_plot.__version__`).

## Development Commands

```bash
# Environment (installs the package editable with dev+docs extras)
conda env create -f environment.yml
conda activate asp_plot
pre-commit install                      # REQUIRED for development

# Tests
pytest                                  # all tests; add -s to see print output
pytest tests/test_stereo.py::test_name  # one file / one test

# Lint/format (pre-commit runs these on commit; manually:)
pre-commit run --all-files              # black + flake8 + isort (profile: black)

# Rebuild after changing CLI tools or entry points
pip install -e ".[dev]"
```

Flake8 config is in `.flake8` (extends ignore: E203, E701); pre-commit further ignores E501, E722, E207.

To build the docs locally (Sphinx + MyST; hosted on ReadTheDocs, auto-built on push to `main`):

```bash
# One-time: copy notebooks, reports, and figures for local preview
mkdir -p docs/examples/notebooks && cp notebooks/**/*.ipynb docs/examples/notebooks/
mkdir -p docs/_static/reports && cp reports/*.pdf docs/_static/reports/
mkdir -p docs/_extra/examples/figures && cp notebooks/figures/* docs/_extra/examples/figures/

sphinx-autobuild docs docs/_build/html --open-browser   # or sphinx-build for a one-off
```

## Gotchas

- **ASP tools must be on PATH** for the workflows that wrap them: `stereo`/`parallel_stereo`, `bundle_adjust`, `point2dem`, `pc_align`, `geodiff`, `dg_mosaic` (called via `run_subprocess_command()`).
- **Internet is required** for basemaps (contextily/Esri tiles), ICESat-2 requests (SlideRule), and ESA WorldCover sampling (public AWS S3 COGs). Tests must not depend on the network — basemap fetches are stubbed (#151).
- **Mars altimetry needs the `*_pts_csv.csv`** (with `PLANET_RAD`), never the `*_topo_csv.csv`: MOLA TOPOGRAPHY is referenced to the oblate areoid while ASP DEMs use the spherical IAU datum — a latitude-dependent offset up to ~10 km that pc_align cannot remove. The loader rejects the topo file with an explanatory error.
- **ASP's `mapproject` writes no log file**; its command is reconstructed from output GeoTIFF metadata (`mapproject.py`), not parsed from logs like the other tools.
- **"Vantor" vs "WorldView" naming is deliberate**: Vantor = copyright/attribution (rights-holder), WorldView = sensor/reader identity (`sensors.py`). Don't reconcile them into one name (#137).

## External Data Sources

- **ICESat-2 ATL06-SR** (Earth): requested through the SlideRule API; results cached as parquet next to the report.
- **LOLA (Moon) / MOLA (Mars)**: async queries to the ODE GDS REST API via the `request_planetary_altimetry` CLI — the user gets a download link by email, then passes the CSV to `asp_plot --altimetry_csv`. Coordinates are east-positive 0–360 longitude, planetocentric latitude.

## Testing

Tests are in `tests/` with sample data in `tests/test_data/` (synthetic rasters, XML camera files, BA CSVs, ICESat-2 parquet, pc_align outputs, jitter data). Most modules have a matching `tests/test_<module>.py`; `test_imports.py` verifies everything imports. Example notebooks in `notebooks/` are organized by sensor (WorldView, ASTER, LRO_NAC, Mars_MGS, Mars_MRO) — see ARCHITECTURE.md for what each demonstrates.

## Versioning and Release Process

Follow semantic versioning. To release:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with version and date
3. Merge to `main`

The rest is automated: `release.yml` detects the version bump, creates a GitHub Release + tag, and publishes to PyPI via OIDC trusted publishing; conda-forge's autotick bot then opens a feedstock PR. PyPI trusted publishing and the conda-forge feedstock are already configured (reference recipe in `conda-forge-recipe/meta.yaml`).

## Common File Patterns

ASP output files follow specific naming patterns (find them with the `glob_file()` utility):
- DEMs: `*-DEM.tif` or `*_dem.tif`
- Disparity: `*-F.tif`
- Match files: `*.match`
- Bundle adjust residuals: `*-initial_residuals_pointmap.csv`, `*-final_residuals_pointmap.csv`
- Log files: `*log-bundle_adjust*.txt`, `*log-stereo*.txt`, `*log-point2dem*.txt`
