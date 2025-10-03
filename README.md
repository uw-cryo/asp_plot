# asp_plot

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14263122.svg)](https://doi.org/10.5281/zenodo.14263122)

Scripts and notebooks to visualize output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

## Motivation

Our objective is to release a modular Python package with a command-line interface (CLI) that can be run automatically on an ASP output directory to prepare a set of standard diagnostic plots, publication-quality output figures, and a pdf report with relevant information, similar to the reports prepared by many commercial SfM software packages (e.g., Agisoft Metashape, Pix4DMapper).


## Status

As of version 1.0.0, ASP Plot provides a stable set of tools for visualizing ASP processing results.
The package follows semantic versioning, and all changes are documented in the [CHANGELOG](CHANGELOG.md).

The directory `original_code/` contains initial notebooks compiled from recent projects using sample stereo images from the Maxar WorldView, Planet SkySat-C and BlackSky Global constellations.

The functionality of these notebooks has been ported to the `asp_plot/` directory, which is the package `asp_plot`.

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
           --reference_dem ref_dem.tif
```

Before that, we recommend running `asp_plot --help` for details (and defaults) of all of the command-line flags:

```
 $ asp_plot --help
Usage: asp_plot [OPTIONS]

  Generate a comprehensive report of ASP processing results.

  Creates a series of diagnostic plots for stereo processing, bundle adjustment,
  ICESat-2 comparisons, and more. All plots are combined into a single PDF report
  with processing parameters and summary information.

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
  --dem_filename TEXT             Optional DEM filename in the stereo
                                  directory. Default: None, which will search
                                  for the *-DEM.tif file in the stereo
                                  directory. Specify it as the basename with
                                  extension, e.g. my-custom-dem-name.tif.
  --dem_gsd TEXT                  Optional ground sample distance of the DEM.
                                  Default: None, which will search for the
                                  *-DEM.tif file in the stereo directory. If
                                  there is a GSD in the name of the file,
                                  specify it here as a float or integer, e.g.
                                  1, 1.5, etc.
  --map_crs TEXT                  Projection for ICESat and bundle adjustment
                                  plots. As EPSG:XXXX. Default: None, which
                                  will use the projection of the ASP DEM, and
                                  fall back on EPSG:4326 if not found.
  --reference_dem TEXT            Optional reference DEM used in ASP
                                  processing. No default. If not supplied, the
                                  logs will be examined to find it. If not
                                  found, no difference plots will be
                                  generated.
  --add_basemap BOOLEAN           If True, add a basemaps to the figures,
                                  which requires internet connection. Default:
                                  True.
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
  --help                          Show this message and exit.
```

### Running without internet connection

If you add these two flags as `False` to the `asp_plot` command, you can run it without internet connection:

```
--add_basemap False --plot_icesat False
```

Otherwise, basemaps will be fetched using contextly and ICESat-2 data will be fetched by SlideRule.

## CLI usage: `stereo_geom`

The `stereo_geom` command-line tool creates visualizations of stereo geometry for satellite imagery based on the XML camera files. It produces a combined plot with a skyplot showing satellite viewing angles and a map view showing the footprints and satellite positions.

At its simplest, you can run:

```
$ stereo_geom --directory /path/to/directory/with/xml/files
```

By default, the tool will save the output as `<directory_name>_stereo_geom.png` in the input directory. You can customize the output location and filename:

```
$ stereo_geom --directory /path/to/directory/with/xml/files \
              --output_directory /path/to/save/plots \
              --output_filename custom_output.png
```

The tool can also add a basemap to the map view (requires internet connection):

```
$ stereo_geom --directory /path/to/directory/with/xml/files \
              --add_basemap True
```

For more details on the available options, run:

```
$ stereo_geom --help
Usage: stereo_geom [OPTIONS]

  Generate stereo geometry plots for DigitalGlobe/Maxar XML files. This tool
  creates a skyplot and map visualization of the satellite positions and
  ground footprints.

Options:
  --directory TEXT         Directory containing XML files for stereo geometry
                           analysis. Default: current directory.
  --add_basemap BOOLEAN    If True, add a basemap to the figures, which
                           requires internet connection. Default: True.
  --output_directory TEXT  Directory to save the output plot. Default: Input
                           directory.
  --output_filename TEXT   Filename for the output plot. Default: Directory
                           name with _stereo_geom.png suffix.
  --help                   Show this message and exit.
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
                      --map_crs EPSG:32728
                      --save_dir path/to/save_directory/
```

If a second camera is not supplied, the tool will happily plot just the single camera:

```
$ csm_camera_plot --original_cameras path/to/original_camera_1 \
                      --optimized_cameras path/to/optimized_camera_1 \
                      --map_crs EPSG:32728
                      --save_dir path/to/save_directory/
```

And there are many more options that can also be modified, by examining `csm_camera_plot --help`:

```
 $ csm_camera_plot --help
Usage: csm_camera_plot [OPTIONS]

  Create diagnostic plots for CSM camera model adjustments.

  Analyzes the changes between original and optimized camera models after bundle
  adjustment or jitter correction. Generates plots showing position and angle differences
  along the satellite trajectory, as well as a mapview of the camera footprints.

Options:
  --original_cameras TEXT         Original camera files, supplied as comma
                                  separated list 'path/to/original_camera_1,path/to/original_camera_2'.
                                  No default. Must be supplied.
  --optimized_cameras TEXT        Optimized camera files, supplied as comma
                                  separated list 'path/to/optimized_camera_1,path/to/optimized_camera_2'.
                                  No default. Must be supplied.
  --map_crs TEXT                  UTM EPSG code for map projection. As EPSG:XXXX. If not
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
  --help                          Show this message and exit.
```

## Development

### Install from source

If you instead want to install the source code for e.g. developing the project:

```
$ git clone git@github.com:uw-cryo/asp_plot.git
$ cd asp_plot
$ conda env create -f environment.yml
$ conda activate asp_plot
$ pre-commit install
$ pip install -e .
```

**Please don't miss the `pre-commit install` step**, which does the linting prior to any commits using the `.pre-commit-config.yaml` file that is included in the repo.

If you want to rebuild the package, for instance while testing changes to the CLI tool, then uninstall and reinstall via:

```
$ pip uninstall asp_plot
$ pip install -e .
```

### Run tests

To ensure the install was successful, tests can be run with:

```
$ pytest
```

When you add a new feature, add some test coverage as well.

### Add a feature

Checkout main and pull to get the latest changes:

```
$ git checkout main
$ git pull
```

Create a feature branch:

```
$ git checkout -b my_feature
```

Make as many commits as you like while you work. When you are ready, submit the changes as a pull request.

After some review, you may be asked to add a few tests for the new functionality. Add those in the `tests/` folder, and check that they work with:

```
$ pytest -s
```

When review of the pull request is complete [_squash_ and merge](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/about-merge-methods-on-github#squashing-your-merge-commits) the changes to `main`, combining your commits into a single, descriptive commit of _why_ the changes were made.

### Versioning and CHANGELOG

This project follows [Semantic Versioning](https://semver.org/) which uses a three-part version number: MAJOR.MINOR.PATCH.

- MAJOR: Incompatible API changes
- MINOR: Added functionality in a backwards compatible manner
- PATCH: Backwards compatible bug fixes and minor enhancements

All notable changes are documented in the [CHANGELOG.md](CHANGELOG.md) file in the repository root. When contributing changes, please add an entry to the CHANGELOG.

### Package and upload

Before uploading a new release:

1. Update version in `pyproject.toml` following semantic versioning rules
2. Update the CHANGELOG.md with the new version and date
3. Push changes to `main` - the GitHub Actions workflow will automatically create a release and tag. (Note: The release workflow, `.github/workflows/release.yml`, automatically creates a GitHub release when `pyproject.toml` is updated on the `main` branch.)

Then build and upload the package:

```
rm -rf dist/
python3 -m pip install --upgrade build
python3 -m build
python3 -m pip install --upgrade twine
python3 -m twine upload dist/*
```
