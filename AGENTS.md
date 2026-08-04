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

[pixi](https://pixi.sh) is supported as an alternative to conda (#184), backed by a committed `pixi.lock` so environments can't drift. There is nothing to create or activate — any `pixi run` installs the environment first if it's missing:

```bash
# Install pixi once per machine, then open a new shell so ~/.pixi/bin is on PATH:
#   curl -fsSL https://pixi.sh/install.sh | sh   (or: brew install pixi / conda install -c conda-forge pixi)
pixi run setup       # pre-commit install (also creates the env on first run)
pixi run test        # pytest
pixi run lint        # pre-commit run --all-files
pixi run docs        # stage docs assets + sphinx-autobuild
pixi shell           # interactive shell in the environment
```

Runtime dependencies are declared in three places that must be kept in sync by hand: `pyproject.toml` (source of truth for the released package), `pixi.toml`, and `conda-forge-recipe/meta.yaml`. After editing `pixi.toml`, run `pixi install` and commit the regenerated `pixi.lock` — CI runs the suite with `locked: true`, which fails if the two have drifted.

To build the docs locally (Sphinx + MyST; hosted on ReadTheDocs, auto-built on push to `main`):

```bash
# One-time: copy notebooks, reports, and figures for local preview
mkdir -p docs/examples/notebooks && cp notebooks/**/*.ipynb docs/examples/notebooks/
mkdir -p docs/_static/reports && cp reports/*.pdf docs/_static/reports/
mkdir -p docs/_extra/examples/figures && cp notebooks/figures/* docs/_extra/examples/figures/

sphinx-autobuild docs docs/_build/html --open-browser   # or sphinx-build for a one-off
```

## Gotchas

- **ASP tools must be on PATH** for the workflows that wrap them: `stereo`/`parallel_stereo`, `bundle_adjust`, `point2dem`, `pc_align`, `geodiff`, `dg_mosaic` (called via `run_subprocess_command()`). **Append** the ASP bin directory to PATH rather than prepending: the ASP release bundles its own `python`, which would shadow your environment's interpreter and break imports.
- **Internet is required** for basemaps (contextily/Esri tiles), ICESat-2 requests (SlideRule), and ESA WorldCover sampling (public AWS S3 COGs). Tests must not depend on the network — basemap fetches are stubbed (#151).
- **Mars altimetry needs the `*_pts_csv.csv`** (with `PLANET_RAD`), never the `*_topo_csv.csv`: MOLA TOPOGRAPHY is referenced to the oblate areoid while ASP DEMs use the spherical IAU datum — a latitude-dependent offset up to ~10 km that pc_align cannot remove. The loader rejects the topo file with an explanatory error.
- **ASP's `mapproject` writes no log file**; its command is reconstructed from output GeoTIFF metadata (`mapproject.py`), not parsed from logs like the other tools.
- **Attribution vs sensor naming is deliberate**: copyright/attribution names the rights-holder (`detect_satellite_attribution()` → `"Vantor"` or `"Airbus DS"`), while readers in the `asp_plot/sensors/` package are named for the satellite family (WorldView, Pleiades). Don't reconcile them into one name (#137).
- **Package vs CLI naming is a deliberate split** (v2.0.0, #165): the *package* is `asp_plot` (`import asp_plot`, `pip install asp-plot`), the report *command* is `asp_report` (`asp_plot/cli/asp_report.py`). There is no `asp_plot` console script and no alias — don't "fix" either name to match the other. The other four CLIs (`stereo_geom`, `csm_camera_plot`, `request_planetary_altimetry`, `gallery`) are unprefixed and unchanged.
- **Airbus DIMAP quaternions are scalar-first** (`Q0` = scalar); they are reordered to the scalar-last `q1..q4` layout the roll/pitch/yaw code expects in `PleiadesMetadata.getAtt_df()`. Don't "fix" the reorder.
- **ASP multiview triangulation of mapprojected images needs `ISISROOT`** (observed with ASP 3.8.0-alpha, non-ISIS `-t pleiades` session): the joint triangulation aborts with an uncatchable `Isis::IException` (`$ISISROOT/IsisPreferences was not found`) that surfaces as a generic "Failed to run"/killed job. Workaround: `export ISISROOT=<ASP install root>` (the release bundles `IsisPreferences` there). Pair runs and raw-image multiview runs are unaffected; full write-up in a PR #155 comment.

## External Data Sources

- **ICESat-2 ATL06-SR** (Earth): requested through the SlideRule API; results cached as parquet next to the report.
- **LOLA (Moon) / MOLA (Mars)**: async queries to the ODE GDS REST API via the `request_planetary_altimetry` CLI — the user gets a download link by email, then passes the CSV to `asp_report --altimetry_csv`. Coordinates are east-positive 0–360 longitude, planetocentric latitude.

## Testing

Tests are in `tests/` with sample data in `tests/test_data/` (synthetic rasters, XML camera files, BA CSVs, ICESat-2 parquet, pc_align outputs, jitter data). Most modules have a matching `tests/test_<module>.py`; `test_imports.py` verifies everything imports. Some fixture derivatives (e.g. match-point CSVs next to `.match` files) are gitignored and regenerate during test runs — untracked files appearing under `tests/test_data/` after `pytest` are expected, don't commit them. Example notebooks in `notebooks/` are organized by sensor (WorldView, Pleiades, ASTER, LRO_NAC, Mars_MGS, Mars_MRO) — see ARCHITECTURE.md for what each demonstrates.

## Versioning and Release Process

Follow semantic versioning. To release:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with version and date
3. Merge to `main`

The rest is automated: `release.yml` detects the version bump, creates a GitHub Release + tag, and publishes to PyPI via OIDC trusted publishing; conda-forge's autotick bot then opens a feedstock PR. PyPI trusted publishing and the conda-forge feedstock are already configured (reference recipe in `conda-forge-recipe/meta.yaml`).

**The autotick bot only bumps `version` and `sha256` — it never syncs dependencies or entry points.** So whenever you add/remove a runtime dependency in `pyproject.toml` or add/rename a `[project.scripts]` entry point, the feedstock's `recipe/meta.yaml` must be edited by hand in the same release (`requirements: run:` and `build: entry_points:` + the matching `test: commands:`). Otherwise the conda build *succeeds* and then fails its own test phase, conda-build moves the package to `broken/`, and **nothing is uploaded** — PyPI advances while conda-forge silently stalls on the last good version. This is not hypothetical: adding `pyyaml` in v1.16.0 (#121) went unmirrored and stalled conda-forge at 1.15.1 for five releases (1.16.0 → 1.19.0), with a red ✗ on the feedstock's default branch the whole time. After releasing, check <https://anaconda.org/conda-forge/asp-plot> actually advanced rather than assuming the bot handled it.

## Common File Patterns

ASP output files follow specific naming patterns (find them with the `glob_file()` utility):
- DEMs: `*-DEM.tif` or `*_dem.tif`
- Disparity: `*-F.tif`
- Match files: `*.match`
- Bundle adjust residuals: `*-initial_residuals_pointmap.csv`, `*-final_residuals_pointmap.csv`
- Log files: `*log-bundle_adjust*.txt`, `*log-stereo*.txt`, `*log-point2dem*.txt`

**Multi-view (>2 scene) runs** keep only the joint products (`*-PC.tif`, `*-DEM.tif`, `*-IntersectionErr.tif`) at the stereo-directory top level; the per-pair intermediates live one level down in `<prefix>-pairN/` (`N-L_sub.tif`, `N-R_sub.tif`, `N-D_sub.tif`, the `.match` file, `N-align-{L,R}.txt`, and an `N-stereo.default` config copy naming that pair's images). Discover them with `find_pair_directories()` rather than globbing — code that assumes the flat pair layout silently degrades to "missing files" placeholders (#160).
