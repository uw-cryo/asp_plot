# dem_benchmark

The `dem_benchmark` command-line tool scores many DEMs against one altimetry sample and puts the results side by side. The `asp_report` altimetry pages assess a single DEM; this tool compares several, for example different scene combinations, processing flows (joint multi-view triangulation vs. pairwise stereo merged with `dem_mosaic`), or parameter settings.

Every DEM is scored with the report's recipe — the cached ICESat-2 ATL06-SR parquet replayed, ESA WorldCover water returns dropped (or the groups named with `--filter-out`), residual outliers removed per DEM — and gets:

- **Coverage** inside the common footprint of all the DEMs (percent valid and km²), so runs with different crop windows compare fairly.
- **Triangulation error**, the median and NMAD of the `*-IntersectionErr.tif` from `point2dem --errorimage`, in the table only. A mosaic has none. It is a consistency check on a run (a misaligned camera shows as a jump from centimetres to metres), not a quality ranking: across DEMs of different geometry it runs opposite to accuracy, because a narrow pair's rays intersect precisely at the wrong height, so it is not drawn.
- **Altimetry residuals** (altimetry minus DEM): count, median, NMAD and RMSE, before and after a per-DEM `pc_align --compute-translation-only`. A translation cannot change NMAD, so the difference between the two columns is the bias that alignment removes. The residuals are reported twice: on each DEM's own surviving points (`dh_*` columns) and on the points valid in every DEM (`*_shared_*` columns). The shared-point numbers are the fair comparison and are what the figure shows and the rows are sorted on: each DEM's voids differ and the outlier cut runs per DEM, so a blend that fills voids is otherwise scored on harder ground than a single pair that skips it, and the ranking can change.
- **Bootstrap intervals** on the shared-point median and NMAD, from a paired block bootstrap that resamples whole ICESat-2 beam tracks (or MOLA orbits) rather than points, because residuals along a beam are spatially correlated. Every DEM is evaluated on the same replicates, so the interval on a DEM's NMAD *difference* to the best DEM (`nmad_vs_best_ci_*`) is a paired estimate; a DEM whose difference interval includes zero is not separable from the best. `p_best` is the fraction of replicates on which the DEM ranks first.
- Optionally, the **difference against one candidate** named as the reference.
- The shared-point **median and NMAD by slope class**, with slope taken from the reference DEM of the stereo runs when the logs name one, or from a DEM you name.

```{figure} ../figures/example_dem_benchmark.png
:alt: Six Atlanta DEMs scored against the same ICESat-2 points: coverage, and residual median and NMAD before and after pc_align with bootstrap intervals
:width: 100%

Six same-pass WorldView-2 DEMs of Atlanta — three single pairs at 5°, 22° and 27° convergence, the three pairs merged with `dem_mosaic`, and 3- and 5-scene multi-view runs — scored on the 6 314 ICESat-2 points valid in all six and sorted best-first by post-alignment NMAD. The error bars are 95 % intervals from a bootstrap over whole beam tracks; the five-scene run and the 27° pair are not separable (`≈ best`), every other difference is. From `notebooks/WorldView/worldview_spacenet_atlanta_mvs.ipynb`.
```

## Basic usage

Point the tool at the DEMs and the ICESat-2 parquet cache that a previous `asp_report` run wrote next to its report:

```bash
dem_benchmark stereo_mvs3/run-DEM.tif stereo_mvs5/run-DEM.tif pairwise_mosaic-DEM.tif \
              --parquet atl06sr_all.parquet
```

This prints the stats table and writes `dem_benchmark.png` (summary figure), `dem_benchmark_histogram.png` (overlaid residual histograms) and `dem_benchmark.csv` (one row per DEM) to the working directory. An unlabelled ASP `run-DEM.tif` is labelled by its folder; any other DEM by its filename. Give your own labels as `LABEL=PATH`:

```bash
dem_benchmark "MVS 3-scene=stereo_mvs3/run-DEM.tif" \
              "3 pairs + mosaic=pairwise_mosaic-DEM.tif" \
              "pair 13-16 (5.1°)=stereo_pair_13_16/run-DEM.tif" \
              --parquet atl06sr_all.parquet \
              --title "Atlanta WV2: same-pass scene combinations"
```

`pc_align` products go under `<directory>/dem_benchmark/<label>/`, never into the DEMs' own folders, and are reused on a re-run. Skip alignment with `--no-pc-align`. The bootstrap runs 1000 replicates by default; `--n-bootstrap 0` skips it.

## Choosing the land cover to drop

By default only water returns are dropped, as in the report. Repeat `--filter-out` to drop several ESA WorldCover groups (`water`, `trees`, `low_vegetation`, `built_up`, `snow_ice`), or pass `--filter-out none` to keep every return. Giving the option replaces the default rather than adding to it: `--filter-out trees` alone drops trees and keeps water, so name `water` as well to keep dropping it. `none` cannot be combined with a group.

```bash
dem_benchmark stereo_mvs3/run-DEM.tif stereo_mvs5/run-DEM.tif \
              --parquet atl06sr_all.parquet \
              --filter-out water --filter-out trees
```

Dropping water and trees is the stable-terrain choice. Years usually separate the imagery from the ICESat-2 passes and canopy height changes over that time, and in a forest a stereo DSM and a 40 m ICESat-2 segment do not measure the same surface, so the residuals there say more about the reference than about the DEM. The filter is applied to the altimetry sample before any DEM is scored, so every DEM is still compared on the same points. A choice other than the default is printed in the figure subtitle.

## Comparing DEMs to one of them

Name one candidate as the reference to add its difference against every other DEM (`vs_ref_median_m`, `vs_ref_nmad_m`):

```bash
dem_benchmark "MVS 5-scene=stereo_mvs5/run-DEM.tif" "MVS 3-scene=stereo_mvs3/run-DEM.tif" \
              --parquet atl06sr_all.parquet --reference "MVS 5-scene"
```

## Accuracy by slope

On steep sites the useful question is often how accuracy changes with slope. By default (`--slope-dem auto`) slope is taken from the reference DEM of the stereo runs, the one `asp_report` finds: the DEM the images were mapprojected onto, named in the stereo logs beside each DEM. A DEM with no logs beside it, such as a lidar DTM, is ignored for this. The breakdown is skipped, with the reason printed, when no log names a reference DEM, when the logs name different ones, or when the file does not exist.

To choose the DEM yourself, pass `--slope-dem` the label of one of the candidates or a path, such as a Copernicus DEM or a lidar DTM. `--slope-dem none` turns the breakdown off.

```bash
dem_benchmark stereo_mvs3/run-DEM.tif stereo_mvs5/run-DEM.tif \
              --parquet atl06sr_all.parquet \
              --slope-dem cop30.tif --slope-bins 0,5,10,20,30,90
```

Slope is computed as `gdaldem slope` does (Horn's method, from the 3 × 3 pixels around each point) and sampled once per point, so every DEM is binned the same way. Taking slope from each candidate instead would put the same point in different classes for different DEMs. It would also sort each DEM's own noise into the steep classes, which is the reason the default is the reference DEM and not a candidate. The slope is that of the named DEM at its own resolution: a 30 m Copernicus DEM gives the slope of the terrain over about one ICESat-2 segment, a 1 m DSM the slope of roofs and walls. Points in the DEM's edge pixels or next to nodata get no slope and fall in no class. Each class includes its lower edge and excludes its upper one.

This adds `dem_benchmark_slope.png`, the shared-point NMAD of every DEM by class, and `dem_benchmark_by_slope.csv`. For each class the table gives the number of points and of beam tracks, the median and NMAD before and after alignment, and the 95 % interval on the NMAD from the bootstrap rerun within the class. The NMAD table is also printed.

Read the steep classes with two caveats:

- **They are thin.** A few hundred points on a handful of beam tracks is common above 20°, and the intervals show it.
- **NMAD rises with slope for reasons other than the DEM.** A horizontal offset of Δx between a DEM and the altimetry adds about Δx · tan(slope) to a residual: 0.35 m at 5° and 1.5 m at 20° for a 4 m offset. ICESat-2 geolocation error is such an offset, and so is the horizontal part of each DEM's `pc_align` translation, whose range the figure's subtitle prints. Comparing the breakdown before alignment (`bench.slope_plot(aligned=False)`) shows how much of the steep-class NMAD the alignment contributes.

## Planetary DEMs

For Moon or Mars DEMs pass the LOLA/MOLA CSV from `request_planetary_altimetry` instead of a parquet; the body is detected from the DEMs:

```bash
dem_benchmark run_a/run-DEM.tif run_b/run-DEM.tif --altimetry-csv lola_pts_csv.csv
```

## Reading the figure

- Rows are sorted best-first by post-alignment NMAD on the shared points (pre-alignment with `--no-pc-align`).
- **Coverage** is percent valid inside the common footprint, with the valid km² printed beside it. `--own-extent` scores each DEM over its own footprint instead.
- **dh median / dh NMAD** are altimetry minus DEM on the points valid in every DEM, open marker before `pc_align` and filled after. Translation-only alignment leaves NMAD unchanged, so those markers coincide. The error bar is the 95 % bootstrap interval on the ranked (post-alignment) value, printed in brackets after it. The subtitle states how many points are shared and how many tracks were resampled.
- In the NMAD panel the best DEM is labelled `best`, and every DEM whose paired NMAD difference to it includes zero is labelled `≈ best`. Read that label, not the overlap of the error bars: DEMs scored on the same points share most of their sampling variation, so their intervals can overlap while the difference between them is tight.

## Full options

```
Usage: dem_benchmark [OPTIONS] DEMS...

  Score many DEMs against one altimetry sample.

  DEMS are paths, optionally labelled as LABEL=PATH (e.g.
  "MVS=stereo_mvs3/run-DEM.tif"); an unlabelled ASP run-DEM.tif is labelled by
  its folder. Every DEM gets: coverage inside the common footprint, the median
  triangulation error from its IntersectionErr raster when present (table
  only; it is a consistency check, not a quality ranking), and the altimetry-
  minus-DEM median / NMAD / RMSE before and (unless --no-pc-align) after a
  pc_align translation, on its own points and on the points valid in every
  DEM, with bootstrap intervals on the latter. Writes a one-row-per-DEM
  summary figure, an overlaid residual histogram, and the stats table as CSV.
  When a slope DEM is found or named (--slope-dem), the shared-point NMAD by
  slope class is added as a figure and a second CSV.

Options:
  --parquet TEXT                  ICESat-2 ATL06-SR parquet cache to score
                                  Earth DEMs against (the atl06sr_all.parquet
                                  a previous asp_report run wrote next to its
                                  report, or from Altimetry.request_atl06sr_mu
                                  lti_processing(save_to_parquet=True)). The
                                  same points are replayed for every DEM; no
                                  SlideRule request is made.
  --altimetry-csv TEXT            LOLA/MOLA CSV to score Moon/Mars DEMs
                                  against (see request_planetary_altimetry).
                                  Use instead of --parquet for planetary DEMs.
  --directory TEXT                Working directory. pc_align products and the
                                  translated DEM copies go under
                                  <directory>/dem_benchmark/<label>/, never
                                  into the DEMs' own folders. Default: current
                                  directory.
  --reference TEXT                Label of one of the DEMs to difference the
                                  others against (vs_ref columns of the stats
                                  table). Default: none.
  --no-pc-align                   Skip the per-DEM pc_align translation;
                                  report pre-alignment residuals only.
  --own-extent                    Compute coverage and triangulation-error
                                  statistics over each DEM's own extent
                                  instead of the intersection of all DEM
                                  footprints.
  --filter-out [built_up|low_vegetation|snow_ice|trees|water|none]
                                  ESA WorldCover group to drop from the
                                  ICESat-2 points before differencing; repeat
                                  the option for several. Giving the option
                                  replaces the default, so name water as well
                                  to keep dropping it: '--filter-out water
                                  --filter-out trees' is the stable-terrain
                                  choice when years separate the imagery from
                                  the ICESat-2 passes. 'none' keeps every
                                  return and cannot be combined with a group.
                                  Earth DEMs only. Default: water.
  --n-bootstrap INTEGER           Replicates of the paired block bootstrap
                                  that puts 95 % intervals on the shared-point
                                  median and NMAD of every DEM and on its NMAD
                                  difference to the best DEM (whole ICESat-2
                                  beam tracks, or MOLA orbits, are resampled).
                                  0 skips it. Default: 1000.
  --slope-dem TEXT                DEM to take slope from for an NMAD-by-slope
                                  breakdown. 'auto' uses the reference DEM
                                  named in the stereo logs beside the DEMS
                                  (the one asp_report finds) and skips the
                                  breakdown when there is none or the logs
                                  disagree. Otherwise the label of one of the
                                  DEMS, a path such as a Copernicus DEM or a
                                  lidar DTM, or 'none' for no breakdown. Slope
                                  (Horn's method, as gdaldem) is sampled once
                                  per point, so every DEM is binned the same
                                  way. Writes a <stem>_slope.png figure and a
                                  <stem>_by_slope.csv table.  [default: auto]
  --slope-bins TEXT               Comma-separated slope class edges in
                                  degrees.  [default: 0,5,10,20,30,90]
  --title TEXT                    Figure title. Default: none.
  --output-directory TEXT         Directory for the figure and stats CSV.
                                  Default: --directory.
  --output-filename TEXT          Figure filename; the stats CSV takes the
                                  same name with a .csv extension, and the
                                  residual histogram figure a _histogram
                                  suffix. Default: dem_benchmark.png.
  --help                          Show this message and exit.
```

## Python API

The same functionality is available via the `DEMBenchmark` class, which also keeps one `Altimetry` object per DEM (`bench.altimetry[label]`) so the usual per-DEM figures — `mapview_plot_atl06sr_to_dem()`, `histogram_by_landcover()` — can be drawn for any candidate afterwards:

```python
from asp_plot.dem_benchmark import DEMBenchmark

bench = DEMBenchmark(
    directory="atlanta_mvs",
    dems={
        "MVS 3-scene": "atlanta_mvs/stereo_mvs3/run-DEM.tif",
        "MVS 5-scene": "atlanta_mvs/stereo_mvs5/run-DEM.tif",
        "3 pairs + mosaic": "atlanta_mvs/pairwise_mosaic-DEM.tif",
    },
    parquet="atlanta_mvs/atl06sr_all.parquet",
    reference="MVS 5-scene",
    filter_out=["water", "trees"],     # default "water"; None keeps every return
)
stats = bench.run()                    # one row per DEM
bench.summary_plot(save_dir="atlanta_mvs", fig_fn="dem_benchmark.png")
bench.histogram_plot()
bench.altimetry["MVS 5-scene"].histogram_by_landcover(key="all")

# When a slope DEM is found in the stereo logs (slope_dem="auto", the default)
# or named (slope_dem="cop30.tif", or a candidate's label):
bench.stats_by_slope_df                # one row per DEM and slope class
bench.slope                            # slope in degrees at each shared point
bench.slope_plot(save_dir="atlanta_mvs", fig_fn="dem_benchmark_slope.png")

# The shared point set and the bootstrap replicates are kept for further use
bench.shared_ids                       # point ids valid in every DEM
bench.dh_aligned["MVS 5-scene"].reindex(bench.shared_ids)   # residuals on them
bench.bootstrap["nmad"]                # (replicates x DEMs) NMAD per replicate
not_separable = stats[stats["nmad_vs_best_ci_low_m"] <= 0]["label"]  # includes the best
```
