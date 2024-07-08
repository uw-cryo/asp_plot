# asp_plot

Scripts and notebooks to visualize output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

## Motivation

Our objective is to release a modular Python package with a command-line interface (CLI) that can be run automatically on an ASP output directory to prepare a set of standard diagnostic plots, publication-quality output figures, and a pdf report with relevant information, similar to the reports prepared by many commercial SfM software packages (e.g., Agisoft Metashape, Pix4DMapper).


## Status

This is a work in progress.

The directory `original_code/` contains initial notebooks compiled from recent projects using sample stereo images from the Maxar WorldView, Planet SkySat-C and BlackSky Global constellations. 

The functionality of these notebooks is being ported to the `asp_plot/` directory, which is the package `asp_plot`.


## Installing, testing, and distributing the package

### For Development

If you instead want to install the source code for e.g. developing the project:

```
$ git clone git@github.com:uw-cryo/asp_plot.git
$ cd asp_plot
$ conda env create -f environment.yml
$ conda activate asp_plot
$ pip install -e .
$ python3 setup.py install
```

To ensure the install was successful, tests can be run with:

```
$ pytest
```

### Package and upload

Update version in `pyproject.toml` and `setup.py`, then:

```
$ python3 -m pip install --upgrade build
$ python3 -m build
$ python3 -m pip install --upgrade twine
$ python3 -m twine upload dist/*
```

### Install via pip

```
pip install asp-plot
```

## Notebook example usage

Examples of the modular usage of the package can be found in the `notebooks/` directory.


## CLI usage

A full report and individual plots can be output via the command-line:

```
$ asp_plot --directory ./asp_processing \
           --bundle_adjust_directory ba \
           --stereo_directory stereo \
           --map_crs EPSG:32604 \
           --reference_dem ref_dem.tif \
           --plots_directory asp_plots \
           --report_filename asp_plot_report.pdf
```

Use:

```
$ asp_plot --help
```

for details (and defaults) of the command-line flags.
