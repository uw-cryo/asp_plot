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

`asp_plot` reads the same satellite camera metadata the Stereo Pipeline itself
does, so a pair ASP can process is a pair `asp_plot` can plot the geometry of:

- **Earth**: WorldView / GeoEye / QuickBird / IKONOS, the Airbus DIMAP family
  (Pléiades 1A/1B and Neo, SPOT 5 and 6/7, PeruSat-1), ASTER, and RPC-only
  products such as Cartosat-1 and Deimos
- **Lunar**: Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC)
- **Mars**: Mars Reconnaissance Orbiter CTX and HiRISE, Mars Global Surveyor MOC

Planetary sensors are handled through their CSM model states
(`csm_camera_plot`) rather than the camera-metadata readers. The one gap is
ASP's `pinhole`/`opticalbar` sessions — historical aerial and declassified film
imagery, which carry no satellite geometry to plot.

Reports work for **any** ASP processing directory regardless: when a sensor's
camera metadata is not supported, the stereo geometry section is skipped and
everything else still renders.

Two of these sensors have no geometry to parse — ASTER camera files record only
look vectors, and RPC-only products are nothing but a camera model — so theirs
is *derived* instead, and validated against published or vendor-reported
geometry. For RPC-only products, point `stereo_geom` at the images themselves
rather than at XMLs, since that is where the camera model lives.

A few readers are written from ASP's reader spec rather than validated against a
real delivery: SPOT 6/7, PeruSat-1, SPOT 5, ALOS PRISM, and Pléiades 1A/1B
attitude. Each warns once when it parses a scene. If you have data for one, a
[report](https://github.com/uw-cryo/asp_plot/issues/new) is very welcome — it is
what turns a spec-only reader into a validated one.

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
