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

A full report can be output via the command-line. At its simplest, you can run:

```
$ asp_plot --directory ./ \
           --stereo_directory stereo
```

with only the directory where the ASP processing was done (`--directory`) and the subdirectory inside of that where the stereo files were output (`--stereo`). The reference DEM used in ASP processing will also be searched for in the logs, and used for difference maps if found.

If you also ran bundle adjustment and/or would like to specify a reference DEM to use for plotting (rather than searching the logs):

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
  --directory TEXT                Required directory of ASP processing with
                                  scenes and sub-directories for stereo and
                                  optionally bundle adjustment. Default:
                                  current directory.
  --bundle_adjust_directory TEXT  Optional directory of bundle adjustment
                                  files. If expected *residuals_pointmap.csv
                                  files are not found in the supplied
                                  directory, no bundle adjustment plots will
                                  be generated. Default: None.
  --stereo_directory TEXT         Required directory of stereo files. Default:
                                  stereo.
  --map_crs TEXT                  Projection for ICESat and bundle adjustment
                                  plots. Default: None.
  --reference_dem TEXT            Optional reference DEM used in ASP
                                  processing. No default. If not supplied, the
                                  logs will be examined to find it. If not
                                  found, no difference plots will be
                                  generated.
  --add_basemap BOOLEAN           If True, add a contextily basemap to the
                                  figure, which requires internet connection.
                                  Default: True.
  --plot_icesat BOOLEAN           If True, plot an ICESat-2 difference plot
                                  with the DEM result. This requires internet
                                  connection to pull ICESat data. Default:
                                  True.
  --report_filename TEXT          PDF file to write out for report into the
                                  processing directory supplied by
                                  --directory. Default: Directory name of ASP
                                  processing.
  --report_title TEXT             Title for the report. Default: Directory
                                  name of ASP processing.
```

## CLI usage: `csm_camera_plot`

The `csm_camera_plot` command-line tool is a wrapper for outputting a summary plot after running tools like `bundle_adjust` and `jitter_solve`. The inputs must be [CSM camera files](https://stereopipeline.readthedocs.io/en/stable/examples/csm.html). Currently, this tool only supports CSM linescan cameras, such as those from WorldView satellites.

At its simplest it can be run like:

```
$ csm_camera_plot --original_cameras path/to/original_camera_1,path/to/original_camera_2 \
                      --optimized_cameras path/to/optimized_camera_1,path/to/optimized_camera_2
```

But, for more meaningful positions we at least recommend specifying a `map_crs` UTM EPSG code, and a directory to save the output figure to:

```
$ csm_camera_plot --original_cameras path/to/original_camera_1,path/to/original_camera_2 \
                      --optimized_cameras path/to/optimized_camera_1,path/to/optimized_camera_2 \
                      --map_crs 32728
                      --save_dir path/to/save_directory/
```

If a second camera is not supplied, the tool will happily plot just the single camera:

```
$ csm_camera_plot --original_cameras path/to/original_camera_1 \
                      --optimized_cameras path/to/optimized_camera_1 \
                      --map_crs 32728
                      --save_dir path/to/save_directory/
```

And there are many more options that can also be modified, by examining `csm_camera_plot --help`:

```
 $ csm_camera_plot --help
Usage: csm_camera_plot [OPTIONS]

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
                                  output. Default: None.
  --trim BOOLEAN                  Trim the beginning and end of the positions
                                  plotted to the first and last camera image
                                  lines. Default: True.
  --shared_scales BOOLEAN         If True, the position and angle difference
                                  scales are shared between for each camera.
                                  Default: False.
  --log_scale_positions BOOLEAN   If True, the position difference scales are
                                  log scaled. Default: False.
  --log_scale_angles BOOLEAN      If True, the angle difference scales are log
                                  scaled. Default: False.
  --upper_magnitude_percentile INTEGER
                                  Percentile to use for the upper limit of the
                                  mapview colorbars. Default: 95.
  --figsize TEXT                  Figure size as width,height. Default: 20,15.
  --save_dir TEXT                 Directory to save the figure. Default: None,
                                  which does not save the figure.
  --fig_fn TEXT                   Figure filename. Default:
                                  csm_camera_plot_summary_plot.png.
  --add_basemap BOOLEAN           If True, add a contextily basemap to the
                                  figure, which requires internet connection.
                                  Default: False.
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
