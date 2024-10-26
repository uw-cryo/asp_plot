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

## Installation

To get started with `asp_plot`, [find the `environment.yml` file here](https://github.com/uw-cryo/asp_plot/blob/main/environment.yml), download it locally, and create a conda environment:

```
$ conda env create -f environment.yml
```

Then activate the environment:

```
$ conda activate asp_plot
```

And finally, install the `asp_plot` package and CLI tools with pip:

```
(asp_plot) $ pip install asp-plot
```

## Notebook example usage

Examples of the modular usage of the package can be found in the [`notebooks/` directory here](https://github.com/uw-cryo/asp_plot/tree/main/notebooks).


## CLI usage: `asp_plot`

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
  --add_basemap BOOLEAN           If True, add a contextily basemap to the
                                  figure, which requires internet connection.
                                  Default: True
  --plot_icesat BOOLEAN           If True, plot an ICESat-2 difference plot
                                  with the DEM result. This requires internet
                                  connection to pull ICESat data. Default:
                                  True
  --report_filename TEXT          PDF file to write out for report into the
                                  processing directory supplied by
                                  --directory. Default: Directory name of ASP
                                  processing
  --report_title TEXT             Title for the report. Default: Directory
                                  name of ASP processing
```

## CLI usage: `camera_optimization`

The `camera_optimization` command-line tool is a wrapper for outputting a summary plot after running tools like `bundle_adjust` and `jitter_solve`.

At its simplest it can be run like:

```
$ camera_optimization --original_cameras path/to/original_camera_1,path/to/original_camera_2 \
                      --optimized_cameras path/to/optimized_camera_1,path/to/optimized_camera_2
```

But, for more meaningful positions we at least recommend specifying a `map_crs` UTM EPSG code, and a directory to save the output figure to:

```
$ camera_optimization --original_cameras path/to/original_camera_1,path/to/original_camera_2 \
                      --optimized_cameras path/to/optimized_camera_1,path/to/optimized_camera_2 \
                      --map_crs 32728
                      --save_dir path/to/save_directory/
```

And there are many more options that can also be modified, by examining `camera_optimization --help`:

```
 $ camera_optimization --help
Usage: camera_optimization [OPTIONS]

Options:
  --original_cameras TEXT         Original camera files, supplied as comma
                                  separated list 'path/to/original_camera_1,path/to/original_camera_2'.
                                  No default. Must be supplied.
  --optimized_cameras TEXT        Optimized camera files, supplied as comma
                                  separated list 'path/to/optimized_camera_1,path/to/optimized_camera_2'.
                                  No default. Must be supplied.
  --map_crs TEXT                  UTM EPSG code for map projection. If not
                                  supplied, the map will be plotted in
                                  original camera coordinates of EPSG:4978
                                  (ECEF).
  --title TEXT                    Optional short title to append to figure
                                  output. Default: None
  --trim BOOLEAN                  Trim the beginning and end of the
                                  geodataframes. Default: False
  --near_zero_tolerance FLOAT     If trim is True, the tolerance for near zero
                                  values of the camera position differences to
                                  trim from the beginning and end. Default:
                                  1e-3
  --trim_percentage INTEGER       If trim is ture, the extra percentage of the
                                  camera positions to trim from the beginning
                                  and end. Default: 5
  --shared_scales BOOLEAN         If True, the position and angle difference
                                  scales are shared between for each camera.
                                  Default: False
  --log_scale_positions BOOLEAN   If True, the position difference scales are
                                  log scaled. Default: False
  --log_scale_angles BOOLEAN      If True, the angle difference scales are log
                                  scaled. Default: False
  --upper_magnitude_percentile INTEGER
                                  Percentile to use for the upper limit of the
                                  mapview colorbars. Default: 95
  --figsize TEXT                  Figure size as width,height. Default: 20,15
  --save_dir TEXT                 Directory to save the figure. Default: None,
                                  which does not save the figure.
  --fig_fn TEXT                   Figure filename. Default:
                                  camera_optimization_summary_plot.png.
  --add_basemap BOOLEAN           If True, add a contextily basemap to the
                                  figure, which requires internet connection.
                                  Default: False
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

Update version in `pyproject.toml` and `setup.py`, then:

```
rm -rf dist/
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload dist/*
```
