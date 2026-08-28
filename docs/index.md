# asp_plot

[![PyPI](https://img.shields.io/pypi/v/asp-plot.svg)](https://pypi.org/project/asp-plot/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/asp-plot.svg)](https://anaconda.org/conda-forge/asp-plot)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14263121.svg)](https://doi.org/10.5281/zenodo.14263121)
[![RTD](https://readthedocs.org/projects/asp-plot/badge/?version=latest)](https://asp-plot.readthedocs.io/en/latest/)

A Python package for visualizing output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

`asp_plot` generates diagnostic plots and comprehensive PDF reports for ASP stereo processing results, similar to reports from commercial SfM software like Agisoft Metashape.

```{figure} figures/example_dem_results.jpg
:alt: Stereo DEM, triangulation intersection error, and difference to the reference DEM for a WorldView-2 multi-view run over Atlanta
:width: 100%

One page of an `asp_report` PDF: the stereo DEM, its triangulation intersection error, and the difference to the reference DEM, for a three-scene WorldView-2 multi-view run over Atlanta. That report, and six more spanning ASTER, the Moon, and Mars, are on the [Example Reports](examples/reports.md) page.
```

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

## What it does

- Stereo DEM processing visualization (hillshades, disparity maps, match points)
- Bundle adjustment analysis (residual maps, histograms)
- CSM camera model comparisons (position/orientation differences)
- ICESat-2 ATL06-SR altimetry comparisons (Earth-based only), with optional automatic `pc_align` refinement and a before/after alignment report
- Stereo geometry visualization from satellite camera metadata
- Comprehensive PDF report generation

## Supported Sensors

`asp_plot` reads the same satellite camera metadata the Stereo Pipeline itself
does, so a pair ASP can process is a pair `asp_plot` can plot the geometry of:

- **Earth**: WorldView / GeoEye / QuickBird / IKONOS, the Airbus DIMAP family
  (Pléiades 1A/1B and Neo, SPOT 5 and 6/7, PeruSat-1), ASTER, and RPC-only
  products such as Cartosat-1 and Deimos
- **Lunar**: Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC)
- **Mars**: Mars Reconnaissance Orbiter CTX and HiRISE, Mars Global Surveyor MOC

Planetary sensors are handled through their CSM model states
(`csm_camera_plot`) rather than the camera-metadata readers.

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
