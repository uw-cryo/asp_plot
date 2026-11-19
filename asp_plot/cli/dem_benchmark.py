import os

import click

from asp_plot.dem_benchmark import SLOPE_BINS, DEMBenchmark
from asp_plot.icesat2_source import WORLDCOVER_GROUPS
from asp_plot.terrain import SLOPE_SCALE_M


@click.command()
@click.argument("dems", nargs=-1, required=True)
@click.option(
    "--parquet",
    prompt=False,
    default=None,
    help="ICESat-2 ATL06-SR parquet cache to score Earth DEMs against (the "
    "atl06sr_all.parquet a previous asp_report run wrote next to its report, "
    "or from Altimetry.request_atl06sr_multi_processing(save_to_parquet=True)). "
    "The same points are replayed for every DEM; no SlideRule request is made.",
)
@click.option(
    "--altimetry-csv",
    prompt=False,
    default=None,
    help="LOLA/MOLA CSV to score Moon/Mars DEMs against (see "
    "request_planetary_altimetry). Use instead of --parquet for planetary DEMs.",
)
@click.option(
    "--directory",
    prompt=False,
    default="./",
    help="Working directory. pc_align products and the translated DEM copies go "
    "under <directory>/dem_benchmark/<label>/, never into the DEMs' own folders. "
    "Default: current directory.",
)
@click.option(
    "--reference",
    prompt=False,
    default=None,
    help="Label of one of the DEMs to difference the others against (vs_ref "
    "columns of the stats table). Default: none.",
)
@click.option(
    "--no-pc-align",
    is_flag=True,
    default=False,
    help="Skip the per-DEM pc_align translation; report pre-alignment residuals only.",
)
@click.option(
    "--own-extent",
    is_flag=True,
    default=False,
    help="Compute coverage and triangulation-error statistics over each DEM's "
    "own extent instead of the intersection of all DEM footprints.",
)
@click.option(
    "--filter-out",
    prompt=False,
    multiple=True,
    default=("water",),
    type=click.Choice(sorted(WORLDCOVER_GROUPS) + ["none"]),
    help="ESA WorldCover group to drop from the ICESat-2 points before "
    "differencing; repeat the option for several. Giving the option replaces "
    "the default, so name water as well to keep dropping it: '--filter-out "
    "water --filter-out trees' is the stable-terrain choice when years "
    "separate the imagery from the ICESat-2 passes. 'none' keeps every return "
    "and cannot be combined with a group. Earth DEMs only. Default: water.",
)
@click.option(
    "--n-bootstrap",
    prompt=False,
    default=1000,
    type=int,
    help="Replicates of the paired block bootstrap that puts 95 % intervals on "
    "the shared-point median and NMAD of every DEM and on its NMAD difference "
    "to the best DEM (whole ICESat-2 beam tracks, or MOLA orbits, are "
    "resampled). 0 skips it. Default: 1000.",
)
@click.option(
    "--slope-dem",
    prompt=False,
    default="auto",
    show_default=True,
    help="DEM to take slope from for a breakdown by slope class. 'auto' uses "
    "the reference DEM named in the stereo logs beside the DEMS (the one "
    "asp_report finds). When there is none or the logs disagree, it uses "
    "the --reference DEM, and skips the breakdown without one. Otherwise the label of one of the DEMS, a path such as "
    "a Copernicus DEM or a lidar DTM, or 'none' for no breakdown. Slope "
    "(Horn's method, as gdaldem) is sampled once per point, so every DEM is "
    "binned the same way. Writes a <stem>_slope.png figure and a "
    "<stem>_by_slope.csv table.",
)
@click.option(
    "--slope-bins",
    prompt=False,
    default=",".join(f"{b:g}" for b in SLOPE_BINS),
    show_default=True,
    help="Comma-separated slope class edges in degrees.",
)
@click.option(
    "--slope-scale",
    prompt=False,
    type=float,
    default=SLOPE_SCALE_M,
    show_default=True,
    help="Length scale of the slope in meters, about one ATL06-SR segment. A "
    "slope DEM with finer pixels is averaged to it first; a coarser one is "
    "used as it is. 0 uses the slope DEM's own resolution.",
)
@click.option(
    "--title",
    prompt=False,
    default=None,
    help="Figure title. Default: none.",
)
@click.option(
    "--output-directory",
    prompt=False,
    default=None,
    help="Directory for the figure and stats CSV. Default: --directory.",
)
@click.option(
    "--output-filename",
    prompt=False,
    default="dem_benchmark.png",
    help="Figure filename; the stats CSV takes the same name with a .csv "
    "extension, and the residual histogram figure a _histogram suffix. "
    "Default: dem_benchmark.png.",
)
def main(
    dems,
    parquet,
    altimetry_csv,
    directory,
    reference,
    no_pc_align,
    own_extent,
    filter_out,
    n_bootstrap,
    slope_dem,
    slope_bins,
    slope_scale,
    title,
    output_directory,
    output_filename,
):
    """
    Score many DEMs against one altimetry sample.

    DEMS are paths, optionally labelled as LABEL=PATH (e.g.
    "MVS=stereo_mvs3/run-DEM.tif"); an unlabelled ASP run-DEM.tif is labelled
    by its folder. Every DEM gets: coverage inside the common footprint, the
    median triangulation error from its IntersectionErr raster when present
    (table only; it is a consistency check, not a quality ranking), and the
    altimetry-minus-DEM median / NMAD / RMSE before and (unless
    --no-pc-align) after a pc_align translation, on its own points and on the
    points valid in every DEM, with bootstrap intervals on the latter. Writes
    a one-row-per-DEM summary figure, an overlaid residual histogram, and the
    stats table as CSV. When a slope DEM is found or named (--slope-dem), the
    shared-point median and NMAD by slope class are added as a figure and a
    second CSV.
    """
    if "none" in filter_out and len(set(filter_out)) > 1:
        raise click.UsageError(
            "--filter-out none keeps every return and cannot be combined with "
            "a land-cover group."
        )
    try:
        slope_bins = [float(b) for b in slope_bins.split(",") if b.strip()]
    except ValueError:
        raise click.BadParameter(
            f"expected comma-separated numbers, got '{slope_bins}'.",
            param_hint="--slope-bins",
        )
    directory = os.path.expanduser(directory)
    if output_directory is None:
        output_directory = directory
    output_directory = os.path.expanduser(output_directory)

    bench = DEMBenchmark(
        directory=directory,
        dems=list(dems),
        parquet=parquet,
        altimetry_csv=altimetry_csv,
        reference=reference,
        aoi=None if own_extent else "intersection",
        filter_out=[g for g in filter_out if g != "none"] or None,
        n_bootstrap=n_bootstrap,
        slope_dem=None if slope_dem == "none" else slope_dem,
        slope_bins=slope_bins,
        slope_scale=slope_scale,
        title=title,
    )
    stats = bench.run(pc_align=not no_pc_align)

    stem = os.path.splitext(output_filename)[0]
    csv_fn = bench.save_stats(os.path.join(output_directory, f"{stem}.csv"))
    bench.summary_plot(save_dir=output_directory, fig_fn=output_filename)
    bench.histogram_plot(save_dir=output_directory, fig_fn=f"{stem}_histogram.png")
    if bench.stats_by_slope_df is not None:
        bench.slope_plot(save_dir=output_directory, fig_fn=f"{stem}_slope.png")

    # The headline columns; the CSV has every column of STATS_COLUMNS.
    shown = stats[
        [
            "label",
            "valid_pct",
            "ie_median_m",
            "n_points",
            "n_shared",
            "dh_shared_median_m",
            "dh_shared_nmad_m",
            "dh_aligned_shared_median_m",
            "dh_aligned_shared_nmad_m",
            "nmad_ci_low_m",
            "nmad_ci_high_m",
            "p_best",
        ]
    ]
    shown = shown.dropna(axis=1, how="all")
    print("\n" + shown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    outputs = (
        f"\nSummary figure: {os.path.join(output_directory, output_filename)}"
        f"\nHistogram:      {os.path.join(output_directory, stem + '_histogram.png')}"
        f"\nStats table:    {csv_fn}"
    )
    by_slope = bench.stats_by_slope_df
    if by_slope is not None:
        stage = (
            "dh_aligned_shared"
            if by_slope["dh_aligned_shared_nmad_m"].notna().any()
            else "dh_shared"
        )
        counts = by_slope.drop_duplicates("slope_bin").set_index("slope_bin")[
            "n_points"
        ]
        for column in (f"{stage}_median_m", f"{stage}_nmad_m"):
            table = by_slope.pivot(index="label", columns="slope_bin", values=column)
            table = table.reindex(
                index=list(stats["label"]), columns=by_slope["slope_bin"].unique()
            )
            table = table.map(lambda v: f"{v:.2f}")
            table.loc["n points"] = counts.astype(int).astype(str)
            print(f"\n{column} by slope class (slope from {bench.slope_dem_name}):")
            print(table.to_string())
        outputs += (
            f"\nSlope figure:   {os.path.join(output_directory, stem + '_slope.png')}"
            f"\nSlope table:    {bench.slope_csv_path(csv_fn)}"
        )
    print(outputs + "\n")


if __name__ == "__main__":
    main()
