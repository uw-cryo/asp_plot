# asp_plot

The main CLI tool generates a comprehensive PDF report of ASP processing results.

## Basic usage

At its simplest, run from the ASP processing directory:

```bash
asp_plot --directory ./ \
         --stereo_directory stereo
```

This requires only the directory where ASP processing was done (`--directory`) and the subdirectory containing stereo outputs (`--stereo_directory`). The reference DEM used in processing is automatically searched for in the logs and used for difference maps if found.

## With bundle adjustment and reference DEM

```bash
asp_plot --directory ./ \
         --bundle_adjust_directory ba \
         --stereo_directory stereo \
         --reference_dem ref_dem.tif
```

## Running without internet

If you don't have internet access, disable basemap and ICESat-2 fetching:

```bash
asp_plot --directory ./ \
         --stereo_directory stereo \
         --add_basemap False \
         --plot_icesat False
```

Otherwise, basemaps are fetched using contextily and ICESat-2 data is fetched via SlideRule.

## Full options

```
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
  --plot_geometry BOOLEAN         If True, plot the stereo geometry. Default:
                                  True.
  --subset_km FLOAT               Size in km of the subset to plot for the
                                  detailed hillshade. Default: 1 km.
  --report_filename TEXT          PDF file to write out for report into the
                                  processing directory supplied by
                                  --directory. Default: Directory name of ASP
                                  processing.
  --report_title TEXT             Title for the report. Default: Directory
                                  name of ASP processing.
  --help                          Show this message and exit.
```

## Files needed from ASP processing

During the `stereo` or `parallel_stereo` steps, add this flag to retain the files needed for plotting:

```
--keep-only '.mask .txt .exr .match -L.tif -L_sub.tif -R_sub.tif -D_sub.tif -D.tif -RD.tif -F.tif -PC.tif'
```
