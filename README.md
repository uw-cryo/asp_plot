# asp_plot

Scripts and notebooks to visualize output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

## Motivation

Our objective is to release a modular Python package with a command-line interface (CLI) that can be run automatically on an ASP output directory to prepare a set of standard diagnostic plots, publication-quality output figures, and a pdf report with relevant information, similar to the reports prepared by many commercial SfM software packages (e.g., Agisoft Metashape, Pix4DMapper).


## Status

This is a work in progress.

The directory `original_code/` contains initial notebooks compiled from recent projects using sample stereo images from the Maxar WorldView, Planet SkySat-C and BlackSky Global constellations.

The functionality of these notebooks is being ported to the `asp_plot/` directory, which is the package `asp_plot`.

## Files you will need from ASP processing

During the `stereo` or `parallel_stereo` steps at the heart of the Ames Stereo Pipeline, you can add this to the command to ensure that the files needed for plotting are retained, and files that are not required are cleaned up:

```
--keep-only '.mask .txt .exr .match -L.tif -L_sub.tif -R_sub.tif -D_sub.tif -D.tif -RD.tif -F.tif -PC.tif'
```

Not all of those files are used in the plotting, but all are useful for re-processing and detailed analyses.

## Install via pip

```
pip install asp-plot
```

## Notebook example usage

Examples of the modular usage of the package can be found in the `notebooks/` directory.


## CLI usage

A full report and individual plots can be output via the command-line:

```
$ asp_plot --directory ./ \
           --bundle_adjust_directory ba \
           --stereo_directory stereo \
           --map_crs EPSG:32604 \
           --reference_dem ref_dem.tif
```

Before that, we recommend running `asp_plot --help` for details (and defaults) of all of the command-line flags:

```
 $ asp_plot --help
Usage: asp_plot [OPTIONS]

Options:
  --directory TEXT                Directory of ASP processing with scenes and
                                  sub-directories for bundle adjustment and
                                  stereo. Default: current directory
  --bundle_adjust_directory TEXT  Directory of bundle adjustment files.
                                  Default: ba
  --stereo_directory TEXT         Directory of stereo files. Default: stereo
  --map_crs TEXT                  Projection for bundle adjustment plots.
                                  Default: EPSG:4326
  --reference_dem TEXT            Reference DEM used in ASP processing. No
                                  default. Must be supplied.
  --plots_directory TEXT          Directory to put output plots. Default:
                                  asp_plots
  --report_filename TEXT          PDF file to write out for report into the
                                  processing directory supplied by
                                  --directory. Default: Directory name of ASP
                                  processing
  --report_title TEXT             Title for the report. Default: Directory
                                  name of ASP processing
```


## Development

### Install from source

If you instead want to install the source code for e.g. developing the project:

```
$ git clone git@github.com:uw-cryo/asp_plot.git
$ cd asp_plot
$ conda env create -f environment.yml
$ conda activate asp_plot
$ pip install -e .
$ python3 setup.py install
```

### Run tests

To ensure the install was successful, tests can be run with:

```
$ pytest
```

When you add a new feature, add some test coverage as well.

### Package and upload

```
$ rm -rf dist/
```

Then update version in `pyproject.toml` and `setup.py`, then:

```
$ python3 -m pip install --upgrade build
$ python3 -m build
$ python3 -m pip install --upgrade twine
$ python3 -m twine upload dist/*
```
