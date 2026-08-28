# dem_benchmark

The `dem_benchmark` command-line tool scores many DEMs against one altimetry sample. The `asp_report` altimetry pages assess *one* DEM; this tool answers the question that needs *many* — which scene combination, which processing flow (joint multi-view triangulation vs. pairwise stereo merged with `dem_mosaic`), which parameter setting produces the best DEM — by scoring every candidate against exactly the same ICESat-2 (or LOLA/MOLA) points and putting the results side by side.

Every DEM is scored with the same recipe the report uses for a single DEM: the cached ATL06-SR parquet is replayed (no SlideRule request), water returns are dropped using the ESA WorldCover classes stored in that cache, and residual outliers beyond 3σ are removed per DEM. For each DEM the tool reports:

- **Coverage** inside a common area of interest — by default the intersection of all the DEM footprints, so runs with different crop windows compare fairly — as percent valid and km².
- **Triangulation error**, the median and NMAD of the `*-IntersectionErr.tif` that `point2dem --errorimage` writes next to `*-DEM.tif`. A mosaic has none; that blank row is itself a finding.
- **Altimetry residuals** (altimetry minus DEM): point count, median, NMAD and RMSE, both as produced and after a per-DEM `pc_align --compute-translation-only`, with the translation it applied. A translation cannot change NMAD, so that column separates bias (removable) from noise (not).
- Optionally, each DEM's **difference against one of the candidates** named as the reference.

```{figure} ../figures/example_dem_benchmark.png
:alt: Six Atlanta DEMs scored against the same ICESat-2 points: coverage, triangulation error, and residual median and NMAD before and after pc_align
:width: 100%

Six same-pass WorldView-2 DEMs of Atlanta — three single pairs at 5°, 22° and 27° convergence, the three pairs merged with `dem_mosaic`, and 3- and 5-scene multi-view runs — scored against one ICESat-2 sample, sorted best-first by post-alignment NMAD. The 5° pair is what drags the mosaic down; the multi-view runs win on bias rather than spread. From `notebooks/WorldView/worldview_spacenet_atlanta_mvs.ipynb`.
```

## Basic usage

Point the tool at the DEMs and the ICESat-2 parquet cache a previous `asp_report` run wrote next to its report (or that `Altimetry.request_atl06sr_multi_processing(save_to_parquet=True)` saved):

```bash
dem_benchmark stereo_mvs3/run-DEM.tif stereo_mvs5/run-DEM.tif pairwise_mosaic-DEM.tif \
              --parquet atl06sr_all.parquet
```

This writes `dem_benchmark.png` (the summary figure), `dem_benchmark_histogram.png` (overlaid residual histograms) and `dem_benchmark.csv` (the full stats table, one row per DEM) into the working directory, and prints the table.

An unlabelled ASP `run-DEM.tif` is labelled by its folder (`stereo_mvs3`); any other DEM by its filename. Give your own labels as `LABEL=PATH`:

```bash
dem_benchmark "MVS 3-scene=stereo_mvs3/run-DEM.tif" \
              "3 pairs + mosaic=pairwise_mosaic-DEM.tif" \
              "pair 13-16 (5.1°)=stereo_pair_13_16/run-DEM.tif" \
              --parquet atl06sr_all.parquet \
              --title "Atlanta WV2: same-pass scene combinations"
```

## Where the pc_align products go

`pc_align` is run once per DEM — its log, transform and the translated DEM copy land under `<directory>/dem_benchmark/<label>/`, never inside the DEMs' own folders, so scoring never litters a stereo run. Existing products there are reused, so a re-run is instant and works offline. Skip alignment entirely with `--no-pc-align`, or if `pc_align` is not on your `PATH` (the tool then reports pre-alignment residuals and says so).

```bash
dem_benchmark stereo_*/run-DEM.tif --parquet atl06sr_all.parquet \
              --directory benchmark_runs --output-directory figures
```

## Comparing DEMs to one of them

Name one candidate as the reference to add its difference against every other DEM (`vs_ref_median_m`, `vs_ref_nmad_m`, DEM minus reference):

```bash
dem_benchmark "MVS 5-scene=stereo_mvs5/run-DEM.tif" "MVS 3-scene=stereo_mvs3/run-DEM.tif" \
              --parquet atl06sr_all.parquet --reference "MVS 5-scene"
```

## Planetary DEMs

For Moon or Mars DEMs pass the LOLA/MOLA CSV from `request_planetary_altimetry` instead of a parquet; the body is detected from the DEMs:

```bash
dem_benchmark run_a/run-DEM.tif run_b/run-DEM.tif --altimetry-csv lola_pts_csv.csv
```

## Reading the figure

- Rows are sorted best-first by post-alignment NMAD (pre-alignment when `--no-pc-align`).
- **Coverage** bars are percent valid inside the common AOI; the km² printed next to each is the valid area. Use `--own-extent` to score each DEM over its own footprint instead (not comparable across crops, but useful for a single-run sanity check).
- **IntersectionErr** is the median triangulation error with the NMAD in parentheses. Note that a narrow-convergence pair has a *small* intersection error because its rays barely diverge — it is not a quality ranking on its own; read it next to the residual panels.
- **dh median / dh NMAD** show altimetry minus DEM with an open marker before `pc_align` and a filled marker after; the connecting line is the change. Translation-only alignment leaves NMAD unchanged by construction, so those markers coincide.

## Full options

```
Usage: dem_benchmark [OPTIONS] DEMS...

  Score many DEMs against one altimetry sample.

  DEMS are paths, optionally labelled as LABEL=PATH (e.g.
  "MVS=stereo_mvs3/run-DEM.tif"); an unlabelled ASP run-DEM.tif is labelled by
  its folder. Every DEM gets: coverage inside the common footprint, the median
  triangulation error from its IntersectionErr raster when present, and the
  altimetry-minus-DEM median / NMAD / RMSE before and (unless --no-pc-align)
  after a pc_align translation. Writes a one-row-per-DEM summary figure, an
  overlaid residual histogram, and the stats table as CSV.

Options:
  --parquet TEXT           ICESat-2 ATL06-SR parquet cache to score Earth DEMs
                           against (the atl06sr_all.parquet a previous
                           asp_report run wrote next to its report, or from Al
                           timetry.request_atl06sr_multi_processing(save_to_pa
                           rquet=True)). The same points are replayed for
                           every DEM; no SlideRule request is made.
  --altimetry-csv TEXT     LOLA/MOLA CSV to score Moon/Mars DEMs against (see
                           request_planetary_altimetry). Use instead of
                           --parquet for planetary DEMs.
  --directory TEXT         Working directory. pc_align products and the
                           translated DEM copies go under
                           <directory>/dem_benchmark/<label>/, never into the
                           DEMs' own folders. Default: current directory.
  --reference TEXT         Label of one of the DEMs to difference the others
                           against (vs_ref columns of the stats table).
                           Default: none.
  --no-pc-align            Skip the per-DEM pc_align translation; report pre-
                           alignment residuals only.
  --own-extent             Compute coverage and triangulation-error statistics
                           over each DEM's own extent instead of the
                           intersection of all DEM footprints.
  --title TEXT             Figure title. Default: none.
  --output-directory TEXT  Directory for the figure and stats CSV. Default:
                           --directory.
  --output-filename TEXT   Figure filename; the stats CSV takes the same name
                           with a .csv extension, and the residual histogram
                           figure a _histogram suffix. Default:
                           dem_benchmark.png.
  --help                   Show this message and exit.
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
)
stats = bench.run()                    # one row per DEM
bench.summary_plot(save_dir="atlanta_mvs", fig_fn="dem_benchmark.png")
bench.histogram_plot()
bench.altimetry["MVS 5-scene"].histogram_by_landcover(key="all")
```
