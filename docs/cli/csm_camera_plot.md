# csm_camera_plot

The `csm_camera_plot` command-line tool creates diagnostic plots after running tools like `bundle_adjust` and `jitter_solve`. The inputs must be [CSM camera files](https://stereopipeline.readthedocs.io/en/stable/examples/csm.html). Currently, this tool only supports CSM linescan cameras, such as those from WorldView satellites.

## Basic usage

```bash
csm_camera_plot --original_cameras path/to/original_camera_1,path/to/original_camera_2 \
                --optimized_cameras path/to/optimized_camera_1,path/to/optimized_camera_2
```

## With UTM projection and save directory

For more meaningful positions, specify a `map_crs` UTM EPSG code:

```bash
csm_camera_plot --original_cameras path/to/original_camera_1,path/to/original_camera_2 \
                --optimized_cameras path/to/optimized_camera_1,path/to/optimized_camera_2 \
                --map_crs EPSG:32728 \
                --save_dir path/to/save_directory/
```

## Single camera

If a second camera is not supplied, the tool will plot just the single camera:

```bash
csm_camera_plot --original_cameras path/to/original_camera_1 \
                --optimized_cameras path/to/optimized_camera_1 \
                --map_crs EPSG:32728 \
                --save_dir path/to/save_directory/
```

## Full options

```
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
