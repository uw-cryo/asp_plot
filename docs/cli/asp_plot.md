# asp_plot

The main CLI tool generates a comprehensive PDF report of ASP processing results.

## Files needed from ASP processing

During the `stereo` or `parallel_stereo` steps, add this flag to retain the files needed for plotting:

```
--keep-only '.mask .txt .exr .match -L.tif -L_sub.tif -R_sub.tif -D_sub.tif -D.tif -RD.tif -F.tif -PC.tif'
```

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

If you don't have internet access, disable basemap and altimetry fetching:

```bash
asp_plot --directory ./ \
         --stereo_directory stereo \
         --add_basemap False \
         --plot_altimetry False
```

Otherwise, basemaps are fetched using contextily and ICESat-2 data (Earth only) is fetched via SlideRule.

## Planetary altimetry (Moon/Mars)

For planetary DEMs, altimetry data must be requested separately using [`request_planetary_altimetry`](request_planetary_altimetry.md), then passed via `--altimetry_csv`:

```bash
# Step 1: Request data (one-time)
request_planetary_altimetry --dem stereo/output-DEM.tif --email user@example.com

# Step 2: After downloading and unzipping the result
asp_plot --directory ./ \
         --stereo_directory stereo \
         --altimetry_csv /path/to/MolaPEDR_*_topo_csv.csv \
         --add_basemap False \
         --plot_geometry False
```

If `--plot_altimetry` is True (the default) and the DEM is non-Earth but no `--altimetry_csv` is provided, the tool prints instructions and skips altimetry plots.

## Automatic pc_align with ICESat-2

When `--plot_altimetry` and `--pc_align` are both True (the defaults) and the DEM is on Earth, the report appends a `pc_align`-based alignment step after the standard ICESat-2 diagnostics. The aligned DEM is only retained on disk when `pc_align` reduces the median absolute error (`p50`) toward 0 by more than 5% **and** the translation magnitude exceeds the minimum threshold relative to the DEM GSD; otherwise the intermediate DEM file is removed so its presence is a truthy signal that the alignment is worth using.

The report adds:

- An **alignment report page**: the kwargs used for `pc_align`, a one-row horizontal stats table (`p16`/`p50`/`p84` before and after, `N`/`E`/`D` shifts, translation magnitude, all in meters), a description of what each column means, and a status line naming the aligned DEM path or explaining why no DEM was retained.
- On success, three additional full-page figures against the aligned DEM: a pre/post land-cover histogram (shared bin edges, per-landcover stats in stacked text boxes whose outline colors match the bar colors), the full elevation profile, and the best/worst 1 km agreement segments. Segment selection is held fixed so Med/NMAD are directly comparable to the unaligned variants.

Disable with `--pc_align False`. Automatically skipped when `--plot_altimetry` / `--plot_icesat` is False.

## ICESat-2 time filtering

The `--atl06sr_time_range` option controls which ICESat-2 data is requested from the SlideRule API. Requesting fewer granules speeds up processing but may miss useful data.

**`"all"` (default)** requests every ICESat-2 pass over the DEM footprint from mission start (2018-10-14) to present. This is recommended for most surfaces, as the full ~7 years of data provides the largest sample for validation.

**`"auto"`** activates date-buffered filtering. It attempts to detect the scene acquisition date from the stereopair XML metadata and requests data within ±1 year of that date. If no XML metadata is found, it falls back to requesting all data.

**`"START,END"`** (e.g. `"2020-01-01,2024-12-31"`) requests data within an explicit date range.

For areas with known temporal surface change (e.g. ice sheets, glaciers), consider using `"auto"` or an explicit date range to restrict the ICESat-2 data to a time window that matches the DEM acquisition. Seasonal or multi-temporal filtering is also available via the Python API (`predefined_temporal_filter_atl06sr`, `generic_temporal_filter_atl06sr`).

## Full options

```
Usage: asp_plot [OPTIONS]

  Generate a comprehensive report of ASP processing results.

  Creates a series of diagnostic plots for stereo processing, bundle adjustment,
  altimetry comparisons, and more. All plots are combined into a single PDF
  report with processing parameters and summary information.

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
  --map_crs TEXT                  Projection for altimetry and bundle
                                  adjustment plots. As EPSG:XXXX. Default:
                                  None, which will use the projection of the
                                  ASP DEM, and fall back on EPSG:4326 if not
                                  found.
  --reference_dem TEXT            Optional reference DEM used in ASP
                                  processing. No default. If not supplied, the
                                  logs will be examined to find it. If not
                                  found, no difference plots will be
                                  generated.
  --add_basemap BOOLEAN           If True, add basemaps to the figures,
                                  which requires internet connection.
                                  Automatically skipped for planetary DEMs.
                                  Default: True.
  --plot_altimetry BOOLEAN        If True, plot altimetry comparisons
                                  (ICESat-2 for Earth, LOLA for Moon, MOLA
                                  for Mars). For planetary DEMs, requires
                                  --altimetry_csv. Default: True.
  --plot_icesat TEXT              Deprecated: use --plot_altimetry instead.
                                  Kept for backward compatibility.
  --altimetry_csv PATH            Path to a LOLA/MOLA *_topo_csv.csv file
                                  from the ODE GDS API. Required for
                                  planetary altimetry plots. Obtain via:
                                  request_planetary_altimetry.
  --pc_align BOOLEAN              If True and --plot_altimetry is True, run
                                  pc_align against ICESat-2 (Earth only) and
                                  append the alignment-report pages.
                                  Disabled automatically when
                                  --plot_altimetry / --plot_icesat is False.
                                  Default: True.
  --plot_geometry BOOLEAN         If True, plot the stereo geometry. Default:
                                  True.
  --subset_km FLOAT               Size in km of the subset to plot for the
                                  detailed hillshade. Default: 1 km.
  --atl06sr_time_range TEXT       Time range for ICESat-2 ATL06-SR data
                                  requests. "all" for all available data
                                  (mission start to present), or "START,END"
                                  for a custom range (e.g.
                                  "2020-01-01,2024-12-31"), or "auto" for
                                  scene metadata +/- 1 year. Default: all.
  --report_filename TEXT          PDF report filename or path. A bare
                                  filename (e.g. 'report.pdf') is saved in
                                  the stereo directory. A path (e.g.
                                  'reports/report.pdf') is used as-is.
                                  Default: auto-generated from directory
                                  name.
  --report_title TEXT             Title for the report. Default: Directory
                                  name of ASP processing.
  --help                          Show this message and exit.
```

During the `stereo` or `parallel_stereo` steps, add this flag to retain the files needed for plotting:

```
--keep-only '.mask .txt .exr .match -L.tif -L_sub.tif -R_sub.tif -D_sub.tif -D.tif -RD.tif -F.tif -PC.tif'
```
