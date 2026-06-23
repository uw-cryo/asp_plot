import click

from asp_plot.report_pipeline import ReportConfig, run_report


def _reconstruct_command():
    """Rebuild the ``asp_plot --flag ...`` invocation for the report record.

    Only options whose value differs from the Click default are emitted, so the
    recorded command mirrors what the user actually typed.
    """
    import shlex

    click_ctx = click.get_current_context()
    cmd_parts = ["asp_plot"]
    for param in click_ctx.command.params:
        val = click_ctx.params.get(param.name)
        if val is not None and val != param.default:
            cmd_parts.append(f"--{param.name} {shlex.quote(str(val))}")
    return " ".join(cmd_parts)


@click.command()
@click.option(
    "--directory",
    prompt=True,
    default="./",
    help="Required directory of ASP processing with scenes and sub-directories for stereo and optionally bundle adjustment. Default: current directory.",
)
@click.option(
    "--bundle_adjust_directory",
    prompt=False,
    default=None,
    help="Optional directory of bundle adjustment files. If expected *residuals_pointmap.csv files are not found in the supplied directory, no bundle adjustment plots will be generated. Default: None.",
)
@click.option(
    "--stereo_directory",
    prompt=True,
    default="stereo",
    help="Required directory of stereo files. Default: stereo.",
)
@click.option(
    "--dem_filename",
    prompt=False,
    default=None,
    help="Optional DEM filename in the stereo directory. Default: None, which will search for the *-DEM.tif file in the stereo directory. Specify it as the basename with extension, e.g. my-custom-dem-name.tif.",
)
@click.option(
    "--dem_gsd",
    prompt=False,
    default=None,
    help="Optional ground sample distance of the DEM. Default: None, which will search for the *-DEM.tif file in the stereo directory. If there is a GSD in the name of the file, specify it here as a float or integer, e.g. 1, 1.5, etc.",
)
@click.option(
    "--map_crs",
    prompt=False,
    default=None,
    help="Projection for altimetry and bundle adjustment plots. As EPSG:XXXX. Default: None, which will use the projection of the ASP DEM, and fall back on EPSG:4326 if not found.",
)
@click.option(
    "--reference_dem",
    prompt=False,
    default=None,
    help="Optional reference DEM used in ASP processing. No default. If not supplied, the logs will be examined to find it. If not found, no difference plots will be generated.",
)
@click.option(
    "--add_basemap",
    prompt=False,
    default=True,
    help="If True, add a basemaps to the figures, which requires internet connection. Default: True.",
)
@click.option(
    "--plot_altimetry",
    prompt=False,
    default=True,
    help="If True, plot altimetry comparisons (ICESat-2 for Earth, LOLA for Moon, MOLA for Mars). For planetary DEMs, requires --altimetry_csv. Default: True.",
)
@click.option(
    "--plot_icesat",
    prompt=False,
    default=None,
    help="Deprecated: use --plot_altimetry instead. Kept for backward compatibility.",
)
@click.option(
    "--altimetry_csv",
    prompt=False,
    default=None,
    type=click.Path(exists=True),
    help="Path to a LOLA/MOLA *_topo_csv.csv file from the ODE GDS API. Required for planetary altimetry plots. Obtain via: request_planetary_altimetry --dem <dem> --email <email>, then download and unzip the result.",
)
@click.option(
    "--pc_align",
    prompt=False,
    default=True,
    help="If True and --plot_altimetry is True, run pc_align against the reference altimetry (ICESat-2 for Earth, MOLA for Mars, LOLA for Moon) and append the alignment-report pages. Disabled automatically when --plot_altimetry / --plot_icesat is False. Default: True.",
)
@click.option(
    "--plot_geometry",
    prompt=False,
    default=True,
    help="If True, plot the stereo geometry. Default: True.",
)
@click.option(
    "--subset_km",
    prompt=False,
    default=1.0,
    help="Size in km of the subset to plot for the detailed hillshade. Default: 1 km.",
)
@click.option(
    "--atl06sr_time_range",
    prompt=False,
    default="all",
    help='Time range for ICESat-2 ATL06-SR data requests. "all" for all available data (mission start to present), or "START,END" for a custom range (e.g. "2020-01-01,2024-12-31"), or "auto" for scene metadata +/- 1 year. Default: all.',
)
@click.option(
    "--reuse_selections",
    prompt=False,
    default=None,
    type=click.Path(exists=True),
    help="Path to a *_figure_selections.yml written by a previous run. When supplied, replays that run's ICESat-2 points (parquet), profile track, best/worst segments, and detailed-hillshade clip boxes so figures are directly comparable across re-processing runs. Default: None.",
)
@click.option(
    "--report_filename",
    prompt=False,
    default=None,
    help="PDF report filename or path. A bare filename (e.g. 'report.pdf') is saved in the stereo directory. A path (e.g. 'reports/report.pdf' or '/tmp/report.pdf') is used as-is. Default: auto-generated from directory name.",
)
@click.option(
    "--report_title",
    prompt=False,
    default=None,
    help="Title for the report. Default: Directory name of ASP processing.",
)
def main(
    directory,
    bundle_adjust_directory,
    stereo_directory,
    dem_filename,
    dem_gsd,
    map_crs,
    reference_dem,
    add_basemap,
    plot_altimetry,
    plot_icesat,
    altimetry_csv,
    pc_align,
    plot_geometry,
    subset_km,
    atl06sr_time_range,
    reuse_selections,
    report_filename,
    report_title,
):
    """
    Generate a comprehensive report of ASP processing results.

    Creates a series of diagnostic plots for stereo processing, bundle adjustment,
    ICESat-2 comparisons, and more. All plots are combined into a single PDF report
    with processing parameters and summary information.
    """
    config = ReportConfig(
        directory=directory,
        bundle_adjust_directory=bundle_adjust_directory,
        stereo_directory=stereo_directory,
        dem_filename=dem_filename,
        dem_gsd=dem_gsd,
        map_crs=map_crs,
        reference_dem=reference_dem,
        add_basemap=add_basemap,
        plot_altimetry=plot_altimetry,
        plot_icesat=plot_icesat,
        altimetry_csv=altimetry_csv,
        pc_align=pc_align,
        plot_geometry=plot_geometry,
        subset_km=subset_km,
        atl06sr_time_range=atl06sr_time_range,
        reuse_selections=reuse_selections,
        report_filename=report_filename,
        report_title=report_title,
        report_command=_reconstruct_command(),
    )
    run_report(config)


if __name__ == "__main__":
    main()
