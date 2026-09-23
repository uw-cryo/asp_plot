# dem_benchmark

The `dem_benchmark` command-line tool scores many DEMs against one altimetry sample and puts the results side by side. The `asp_report` altimetry pages assess a single DEM; this tool compares several, for example different scene combinations, processing flows (joint multi-view triangulation vs. pairwise stereo merged with `dem_mosaic`), or parameter settings.

Every DEM is scored with the report's recipe — the cached ICESat-2 ATL06-SR parquet replayed, ESA WorldCover water returns dropped, residual outliers removed per DEM — and gets:

- **Coverage** inside the common footprint of all the DEMs (percent valid and km²), so runs with different crop windows compare fairly.
- **Triangulation error**, the median and NMAD of the `*-IntersectionErr.tif` from `point2dem --errorimage`, in the table only. A mosaic has none. It is a consistency check on a run (a misaligned camera shows as a jump from centimetres to metres), not a quality ranking: across DEMs of different geometry it runs opposite to accuracy, because a narrow pair's rays intersect precisely at the wrong height, so it is not drawn.
- **Altimetry residuals** (altimetry minus DEM): count, median, NMAD and RMSE, before and after a per-DEM `pc_align --compute-translation-only`. A translation cannot change NMAD, so the difference between the two columns is the bias that alignment removes. The residuals are reported twice: on each DEM's own surviving points (`dh_*` columns) and on the points valid in every DEM (`*_shared_*` columns). The shared-point numbers are the fair comparison and are what the figure shows and the rows are sorted on: each DEM's voids differ and the outlier cut runs per DEM, so a blend that fills voids is otherwise scored on harder ground than a single pair that skips it, and the ranking can change.
- **Bootstrap intervals** on the shared-point median and NMAD, from a paired block bootstrap that resamples whole ICESat-2 beam tracks (or MOLA orbits) rather than points, because residuals along a beam are spatially correlated. Every DEM is evaluated on the same replicates, so the interval on a DEM's NMAD *difference* to the best DEM (`nmad_vs_best_ci_*`) is a paired estimate; a DEM whose difference interval includes zero is not separable from the best. `p_best` is the fraction of replicates on which the DEM ranks first.
- Optionally, the **difference against one candidate** named as the reference.

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

## Comparing DEMs to one of them

Name one candidate as the reference to add its difference against every other DEM (`vs_ref_median_m`, `vs_ref_nmad_m`):

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
  only; it is a consistency check, not a quality ranking), and the
  altimetry-minus-DEM median / NMAD / RMSE before and (unless --no-pc-align)
  after a pc_align translation, on its own points and on the points valid in
  every DEM, with bootstrap intervals on the latter. Writes a one-row-per-DEM
  summary figure, an overlaid residual histogram, and the stats table as CSV.

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
  --n-bootstrap INTEGER    Replicates of the paired block bootstrap that puts
                           95 % intervals on the shared-point median and NMAD
                           of every DEM and on its NMAD difference to the best
                           DEM (whole ICESat-2 beam tracks, or MOLA orbits,
                           are resampled). 0 skips it. Default: 1000.
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

# The shared point set and the bootstrap replicates are kept for further use
bench.shared_ids                       # point ids valid in every DEM
bench.dh_aligned["MVS 5-scene"].reindex(bench.shared_ids)   # residuals on them
bench.bootstrap["nmad"]                # (replicates x DEMs) NMAD per replicate
not_separable = stats[~(stats["nmad_vs_best_ci_low_m"] > 0)]["label"]
```
