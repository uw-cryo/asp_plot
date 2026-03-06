# asp_plot

[![PyPI](https://img.shields.io/pypi/v/asp-plot.svg)](https://pypi.org/project/asp-plot/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/asp-plot.svg)](https://anaconda.org/conda-forge/asp-plot)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14263121.svg)](https://doi.org/10.5281/zenodo.14263121)
[![Documentation](https://readthedocs.org/projects/asp-plot/badge/?version=latest)](https://asp-plot.readthedocs.io/en/latest/)

A Python package for visualizing output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline). Generates diagnostic plots and comprehensive PDF reports for ASP stereo processing results, similar to reports from commercial SfM software like Agisoft Metashape.

### **[View some example reports](https://asp-plot.readthedocs.io/en/latest/examples/reports.html)**

### **[Full documentation at asp-plot.readthedocs.io](https://asp-plot.readthedocs.io)**

## Installation

```
conda install -c conda-forge asp-plot
```

Or with pip:

```
pip install asp-plot
```

See the [installation guide](https://asp-plot.readthedocs.io/en/latest/installation.html) for more options.

## Quick start

Generate a PDF report from an ASP processing directory:

```
asp_plot --directory ./ --stereo_directory stereo
```

See the [CLI documentation](https://asp-plot.readthedocs.io/en/latest/cli/index.html) for all options and additional tools (`stereo_geom`, `csm_camera_plot`).

## Examples

- [Example reports](https://asp-plot.readthedocs.io/en/latest/examples/reports.html) — PDF reports for WorldView, ASTER, LRO NAC, and Mars sensors
- [Example notebooks](https://asp-plot.readthedocs.io/en/latest/examples/index.html) — Modular usage by sensor type

## Contributing

See the [contributing guide](https://asp-plot.readthedocs.io/en/latest/contributing.html) for development setup, testing, and release process.
