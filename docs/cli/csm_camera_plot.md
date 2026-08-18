# csm_camera_plot

The `csm_camera_plot` command-line tool creates diagnostic plots after running tools like `bundle_adjust` and `jitter_solve`. The inputs must be [CSM camera files](https://stereopipeline.readthedocs.io/en/stable/examples/csm.html). Currently, this tool only supports CSM linescan cameras, such as those from WorldView satellites.

## Basic usage

```bash
csm_camera_plot --original-cameras path/to/original_camera_1,path/to/original_camera_2 \
                --optimized-cameras path/to/optimized_camera_1,path/to/optimized_camera_2
```

## With UTM projection and save directory

For more meaningful positions, specify a `map_crs` UTM EPSG code:

```bash
csm_camera_plot --original-cameras path/to/original_camera_1,path/to/original_camera_2 \
                --optimized-cameras path/to/optimized_camera_1,path/to/optimized_camera_2 \
                --map-crs EPSG:32728 \
                --output-directory path/to/save_directory/
```

## Single camera

If a second camera is not supplied, the tool will plot just the single camera:

```bash
csm_camera_plot --original-cameras path/to/original_camera_1 \
                --optimized-cameras path/to/optimized_camera_1 \
                --map-crs EPSG:32728 \
                --output-directory path/to/save_directory/
```

## Full options

```
Usage: csm_camera_plot [OPTIONS]

  Create diagnostic plots for CSM camera model adjustments.

  Analyzes the changes between original and optimized camera models after
  bundle adjustment or jitter correction. Generates plots showing position and
  angle differences along the satellite trajectory, as well as a mapview of
  the camera footprints.

Options:
  --original-cameras TEXT         Original camera files, supplied as comma
                                  separated list 'path/to/original_camera_1,pa
                                  th/to/original_camera_2'. No default. Must
                                  be supplied.
  --optimized-cameras TEXT        Optimized camera files, supplied as comma
                                  separated list 'path/to/optimized_camera_1,p
                                  ath/to/optimized_camera_2'. No default. Must
                                  be supplied.
  --map-crs TEXT                  UTM EPSG code for map projection. As
                                  EPSG:XXXX. If not supplied, the map will be
                                  plotted in original camera coordinates of
                                  EPSG:4978 (ECEF).
  --title TEXT                    Optional short title to append to figure
                                  output. Default: None.
  --trim / --no-trim              Trim the beginning and end of the positions
                                  plotted to the first and last camera image
                                  lines. Default: True.
  --shared-scales / --no-shared-scales
                                  If True, the position and angle difference
                                  scales are shared between each camera.
                                  Default: False.
  --log-scale-positions / --no-log-scale-positions
                                  If True, the position difference scales are
                                  log scaled. Default: False.
  --log-scale-angles / --no-log-scale-angles
                                  If True, the angle difference scales are log
                                  scaled. Default: False.
  --upper-magnitude-percentile INTEGER
                                  Percentile to use for the upper limit of the
                                  mapview colorbars. Default: 95.
  --figure-size TEXT              Figure size as width,height. Default: 20,15.
  --output-directory TEXT         Directory to save the figure. Default: None,
                                  which does not save the figure.
  --output-filename TEXT          Figure filename. Default:
                                  csm_camera_summary_plot.png.
  --add-basemap / --no-add-basemap
                                  If True, add a contextily basemap to the
                                  figure, which requires internet connection.
                                  Default: False.
  --help                          Show this message and exit.
```
