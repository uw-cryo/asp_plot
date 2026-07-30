# asp_plot

[![PyPI](https://img.shields.io/pypi/v/asp-plot.svg)](https://pypi.org/project/asp-plot/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/asp-plot.svg)](https://anaconda.org/conda-forge/asp-plot)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14263121.svg)](https://doi.org/10.5281/zenodo.14263121)
[![RTD](https://readthedocs.org/projects/asp-plot/badge/?version=latest)](https://asp-plot.readthedocs.io/en/latest/)

A Python package for visualizing output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

`asp_plot` generates diagnostic plots and comprehensive PDF reports for ASP stereo processing results, similar to reports from commercial SfM software like Agisoft Metashape.

## Features

::::{grid} 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install via conda or pip.
:::

:::{grid-item-card} CLI Usage
:link: cli/index
:link-type: doc

Generate reports from the command line.
:::

:::{grid-item-card} Example Reports
:link: examples/reports
:link-type: doc

View PDF reports for different sensors.
:::

:::{grid-item-card} Example Notebooks
:link: examples/index
:link-type: doc

Modular usage examples by sensor type.
:::

:::{grid-item-card} API Reference
:link: autoapi/index
:link-type: doc

Full Python API documentation.
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

Development setup, testing, and releases.
:::

::::

## Supported Sensors

- **Earth-based**: WorldView, Pléiades, ASTER
- **Lunar**: Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC)
- **Mars**: Mars Reconnaissance Orbiter CTX and HiRISE, Mars Global Surveyor MOC

### Stereo-geometry metadata support

PDF reports work for any ASP processing directory — when a sensor's camera
metadata is not supported, the stereo geometry section is skipped gracefully
and everything else still renders. The table below is specifically about the
camera-metadata readers behind the stereo geometry plots (`stereo_geom` and
the report's Stereo Geometry section); see
[uw-cryo/asp_plot#168](https://github.com/uw-cryo/asp_plot/issues/168) for the
expansion roadmap.

| Sensor | ASP session | Camera metadata format | Status |
|---|---|---|---|
| WorldView / GeoEye / QuickBird / IKONOS | `dg` | DigitalGlobe XML | ✅ Supported (validated with real data) |
| Pléiades Neo | `pleiades` | DIMAP v2 (`PNEO_SENSOR`) | ✅ Supported (validated with real data) |
| Pléiades 1A/1B | `pleiades` | DIMAP v2 (`PHR_SENSOR`) | ⚠️ Supported except attitude ([#161](https://github.com/uw-cryo/asp_plot/issues/161)) |
| SPOT 6/7 | `pleiades` | DIMAP v2 (`S6_SENSOR`/`S7_SENSOR`) | 🚧 Planned ([#168](https://github.com/uw-cryo/asp_plot/issues/168)) |
| PeruSat-1 | `perusat` | DIMAP v2 (`PER1_SENSOR`) | 🚧 Planned ([#168](https://github.com/uw-cryo/asp_plot/issues/168)) |
| SPOT 5 | `spot5` | DIMAP v1 | 🚧 Planned ([#168](https://github.com/uw-cryo/asp_plot/issues/168)) |
| ALOS PRISM | `prism` | DIMAP-like (`ALOS`) | 🚧 Planned ([#168](https://github.com/uw-cryo/asp_plot/issues/168)) |
| ASTER | `aster` | `gen_aster` XML | 🚧 Planned, derived geometry ([#175](https://github.com/uw-cryo/asp_plot/issues/175)) |
| Cartosat-1, Deimos, and other RPC-only products | `rpc` | RPC coefficients only | ➖ No trajectory/attitude metadata exists to plot |
| Planetary (LRO NAC, CTX, HiRISE, MOC, ...) | `csm` / ISIS | CSM model state (JSON) | Handled by the CSM camera modules (`csm_camera_plot`), not the sensor readers |

## What it does

- Stereo DEM processing visualization (hillshades, disparity maps, match points)
- Bundle adjustment analysis (residual maps, histograms)
- CSM camera model comparisons (position/orientation differences)
- ICESat-2 ATL06-SR altimetry comparisons (Earth-based only), with optional automatic `pc_align` refinement and a before/after alignment report
- Stereo geometry visualization from satellite XML metadata
- Comprehensive PDF report generation

```{toctree}
:maxdepth: 2
:hidden:

installation
cli/index
examples/index
examples/reports
API Reference <autoapi/index>
contributing
changelog
```
