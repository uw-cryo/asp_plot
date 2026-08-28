import os

import click

from asp_plot.dem_benchmark import DEMBenchmark


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
    title,
    output_directory,
    output_filename,
):
    """
    Score many DEMs against one altimetry sample.

    DEMS are paths, optionally labelled as LABEL=PATH (e.g.
    "MVS=stereo_mvs3/run-DEM.tif"); an unlabelled ASP run-DEM.tif is labelled
    by its folder. Every DEM gets: coverage inside the common footprint, the
    median triangulation error from its IntersectionErr raster when present,
    and the altimetry-minus-DEM median / NMAD / RMSE before and (unless
    --no-pc-align) after a pc_align translation. Writes a one-row-per-DEM
    summary figure, an overlaid residual histogram, and the stats table as CSV.
    """
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
        title=title,
    )
    stats = bench.run(pc_align=not no_pc_align)

    stem = os.path.splitext(output_filename)[0]
    csv_fn = bench.save_stats(os.path.join(output_directory, f"{stem}.csv"))
    bench.summary_plot(save_dir=output_directory, fig_fn=output_filename)
    bench.histogram_plot(save_dir=output_directory, fig_fn=f"{stem}_histogram.png")

    shown = stats.drop(columns=["dem_fn"])
    print("\n" + shown.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(
        f"\nSummary figure: {os.path.join(output_directory, output_filename)}"
        f"\nHistogram:      {os.path.join(output_directory, stem + '_histogram.png')}"
        f"\nStats table:    {csv_fn}\n"
    )


if __name__ == "__main__":
    main()
