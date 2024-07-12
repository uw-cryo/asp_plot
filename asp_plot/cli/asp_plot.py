import os
import glob
import subprocess
import click
import contextily as ctx
from itertools import count
from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.scenes import ScenePlotter, SceneGeometryPlotter
from asp_plot.bundle_adjust import ReadBundleAdjustFiles, PlotBundleAdjustFiles
from asp_plot.stereo import StereoPlotter
from asp_plot.utils import compile_report


@click.command()
@click.option(
    "--directory",
    prompt=True,
    default="./",
    help="Directory of ASP processing with scenes and sub-directories for bundle adjustment and stereo. Default: current directory",
)
@click.option(
    "--bundle_adjust_directory",
    prompt=True,
    default="ba",
    help="Directory of bundle adjustment files. Default: ba",
)
@click.option(
    "--stereo_directory",
    prompt=True,
    default="stereo",
    help="Directory of stereo files. Default: stereo",
)
@click.option(
    "--map_crs",
    prompt=True,
    default="EPSG:4326",
    help="Projection for bundle adjustment plots. Default: EPSG:4326",
)
@click.option(
    "--reference_dem",
    prompt=True,
    default="",
    help="Reference DEM used in ASP processing. Default: ",
)
@click.option(
    "--plots_directory",
    prompt=False,
    default="asp_plots_for_report",
    help="Directory to put output plots. Default: asp_plots",
)
@click.option(
    "--report_filename",
    prompt=False,
    default=None,
    help="PDF file to write out for report into the processing directory supplied by --directory. Default: Directory name of ASP processing",
)
@click.option(
    "--report_title",
    prompt=False,
    default=None,
    help="Title for the report. Default: Directory name of ASP processing",
)
def main(
    directory,
    bundle_adjust_directory,
    stereo_directory,
    map_crs,
    reference_dem,
    plots_directory,
    report_filename,
    report_title,
):
    print(f"\n\nProcessing ASP files in {directory}\n\n")

    plots_directory = os.path.join(directory, plots_directory)
    os.makedirs(plots_directory, exist_ok=True)

    if report_filename is None:
        report_filename = f"asp_plot_report_{os.path.split(directory.rstrip("/\\"))[-1]:}.pdf"
    report_pdf_path = os.path.join(directory, report_filename)

    figure_counter = count(0)

    # Geometry plot
    plotter = SceneGeometryPlotter(directory)

    plotter.dg_geom_plot(
        save_dir=plots_directory, fig_fn=f"{next(figure_counter):02}.png"
    )

    # Scene plot
    plotter = ScenePlotter(directory, stereo_directory, title="Mapprojected Scenes")

    plotter.plot_orthos(
        save_dir=plots_directory, fig_fn=f"{next(figure_counter):02}.png"
    )

    # Bundle adjustment plots
    ba_files = ReadBundleAdjustFiles(directory, bundle_adjust_directory)
    resid_initial_gdf, resid_final_gdf = ba_files.get_initial_final_residuals_gdfs()
    geodiff_initial_gdf, geodiff_final_gdf = ba_files.get_initial_final_geodiff_gdfs()
    resid_mapprojected_gdf = ba_files.get_mapproj_residuals_gdf()

    ctx_kwargs = {
        "crs": map_crs,
        "source": ctx.providers.Esri.WorldImagery,
        "attribution_size": 0,
        "alpha": 0.5,
    }

    plotter = PlotBundleAdjustFiles(
        [resid_initial_gdf, resid_final_gdf],
        lognorm=True,
        title="Bundle Adjust Initial and Final Residuals (Log Scale)",
    )

    plotter.plot_n_gdfs(
        column_name="mean_residual",
        cbar_label="Mean Residual (m)",
        map_crs=map_crs,
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
        **ctx_kwargs,
    )

    plotter.lognorm = False
    plotter.title = "Bundle Adjust Initial and Final Residuals (Linear Scale)"

    plotter.plot_n_gdfs(
        column_name="mean_residual",
        cbar_label="Mean Residual (m)",
        common_clim=False,
        map_crs=map_crs,
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
        **ctx_kwargs,
    )

    plotter = PlotBundleAdjustFiles(
        [resid_mapprojected_gdf],
        title="Bundle Adjust Midpoint distance between\nfinal interest points projected onto reference DEM",
    )

    plotter.plot_n_gdfs(
        column_name="mapproj_ip_dist_meters",
        cbar_label="Interest point distance (m)",
        map_crs=map_crs,
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
        **ctx_kwargs,
    )

    plotter = PlotBundleAdjustFiles(
        [geodiff_initial_gdf, geodiff_final_gdf],
        lognorm=False,
        title="Bundle Adjust Initial and Final Geodiff vs. Reference DEM",
    )

    clim = (
        float(geodiff_initial_gdf["height_diff_meters"].quantile(0.05)),
        float(geodiff_initial_gdf["height_diff_meters"].quantile(0.95)),
    )
    abs_max = max(abs(clim[0]), abs(clim[1]))
    clim = (-abs_max, abs_max)

    plotter.plot_n_gdfs(
        column_name="height_diff_meters",
        cbar_label="Height difference (m)",
        map_crs=map_crs,
        cmap="RdBu",
        clim=clim,
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
        **ctx_kwargs,
    )

    # Stereo plots
    plotter = StereoPlotter(
        directory,
        stereo_directory,
        reference_dem,
        out_dem_gsd=1,
        title="Stereo Match Points",
    )

    plotter.plot_match_points(
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
    )

    plotter.title = "Disparity (pixels)"

    plotter.plot_disparity(
        unit="pixels",
        quiver=True,
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
    )

    plotter.title = "Stereo DEM Results"

    plotter.plot_dem_results(
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
    )

    # Compile report
    processing_parameters = ProcessingParameters(
        directory, bundle_adjust_directory, stereo_directory
    )
    processing_parameters_dict = processing_parameters.from_log_files()

    compile_report(
        plots_directory,
        processing_parameters_dict,
        report_pdf_path,
        report_title=report_title,
    )

    print(f"\n\nReport saved to {report_pdf_path}\n\n")


if __name__ == "__main__":
    main()
