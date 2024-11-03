import os
import shutil
from itertools import count

import click
import contextily as ctx

from asp_plot.altimetry import Altimetry
from asp_plot.bundle_adjust import PlotBundleAdjustFiles, ReadBundleAdjustFiles
from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.scenes import SceneGeometryPlotter, ScenePlotter
from asp_plot.stereo import StereoPlotter
from asp_plot.utils import compile_report


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
    "--map_crs",
    prompt=False,
    default=None,
    help="Projection for ICESat and bundle adjustment plots. Default: None.",
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
    help="If True, add a contextily basemap to the figure, which requires internet connection. Default: True.",
)
@click.option(
    "--plot_icesat",
    prompt=False,
    default=True,
    help="If True, plot an ICESat-2 difference plot with the DEM result. This requires internet connection to pull ICESat data. Default: True.",
)
@click.option(
    "--report_filename",
    prompt=False,
    default=None,
    help="PDF file to write out for report into the processing directory supplied by --directory. Default: Directory name of ASP processing.",
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
    map_crs,
    reference_dem,
    add_basemap,
    plot_icesat,
    report_filename,
    report_title,
):
    print(f"\nProcessing ASP files in {directory}\n")

    plots_directory = os.path.join(directory, "tmp_asp_report_plots/")
    os.makedirs(plots_directory, exist_ok=True)

    if report_filename is None:
        report_filename = (
            f"asp_plot_report_{os.path.split(directory.rstrip('/\\'))[-1]}.pdf"
        )
    report_pdf_path = os.path.join(directory, report_filename)

    figure_counter = count(0)

    if map_crs is None:
        print(
            "\nNo map projection supplied. Defaulting to EPSG:4326. If you want a different projection, supply it with the --map_crs flag.\n"
        )
        map_crs = "EPSG:4326"
        add_basemap = False

    if add_basemap:
        ctx_kwargs = {
            "crs": map_crs,
            "source": ctx.providers.Esri.WorldImagery,
            "attribution_size": 0,
            "alpha": 0.5,
        }
    else:
        ctx_kwargs = {}

    # Stereo plots
    plotter = StereoPlotter(
        directory,
        stereo_directory,
        reference_dem=reference_dem,
        out_dem_gsd=1,
        title="Hillshade with details",
    )

    asp_dem = plotter.dem_fn

    plotter.plot_detailed_hillshade(
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
    )

    plotter.title = "Stereo DEM Results"
    plotter.plot_dem_results(
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

    plotter.title = "Stereo Match Points"
    plotter.plot_match_points(
        save_dir=plots_directory,
        fig_fn=f"{next(figure_counter):02}.png",
    )

    # Scene plot
    plotter = ScenePlotter(directory, stereo_directory, title="Mapprojected Scenes")
    plotter.plot_orthos(
        save_dir=plots_directory, fig_fn=f"{next(figure_counter):02}.png"
    )

    # Geometry plot
    plotter = SceneGeometryPlotter(directory)
    plotter.dg_geom_plot(
        save_dir=plots_directory, fig_fn=f"{next(figure_counter):02}.png"
    )

    # ICESat-2 comparison
    if plot_icesat:
        icesat = Altimetry(dem_fn=asp_dem)

        icesat.pull_atl06sr(
            esa_worldcover=True,
            save_to_parquet=False,
        )

        icesat.filter_atl06sr(
            mask_worldcover_water=True,
            save_to_parquet=False,
            save_to_csv=False,
        )

        icesat.mapview_plot_atl06sr_to_dem(
            title=f"Filtered ICESat-2 minus DEM (n={icesat.atl06sr_filtered.shape[0]})",
            save_dir=plots_directory,
            fig_fn=f"{next(figure_counter):02}.png",
            **ctx_kwargs,
        )

        icesat.histogram(
            title=f"Filtered ICESat-2 minus DEM (n={icesat.atl06sr_filtered.shape[0]})",
            save_dir=plots_directory,
            fig_fn=f"{next(figure_counter):02}.png",
        )

    # Bundle adjustment plots
    if bundle_adjust_directory:
        try:
            ba_files = ReadBundleAdjustFiles(directory, bundle_adjust_directory)
            resid_initial_gdf, resid_final_gdf = (
                ba_files.get_initial_final_residuals_gdfs()
            )
            geodiff_initial_gdf, geodiff_final_gdf = (
                ba_files.get_initial_final_geodiff_gdfs()
            )
            resid_mapprojected_gdf = ba_files.get_mapproj_residuals_gdf()

            plotter = PlotBundleAdjustFiles(
                [resid_initial_gdf, resid_final_gdf],
                lognorm=True,
                title="Bundle Adjust Initial and Final Residuals (Log Scale)",
            )

            plotter.plot_n_gdfs(
                column_name="mean_residual",
                cbar_label="Mean residual (px)",
                map_crs=map_crs,
                save_dir=plots_directory,
                fig_fn=f"{next(figure_counter):02}.png",
                **ctx_kwargs,
            )

            plotter.lognorm = False
            plotter.title = "Bundle Adjust Initial and Final Residuals (Linear Scale)"

            plotter.plot_n_gdfs(
                column_name="mean_residual",
                cbar_label="Mean residual (px)",
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

            plotter.plot_n_gdfs(
                column_name="height_diff_meters",
                cbar_label="Height difference (m)",
                map_crs=map_crs,
                cmap="RdBu",
                symm_clim=True,
                save_dir=plots_directory,
                fig_fn=f"{next(figure_counter):02}.png",
                **ctx_kwargs,
            )
        except ValueError:
            print(
                f"\n\nNo bundle adjustment files found in directory {os.path.join(directory, bundle_adjust_directory):}. If you want bundle adjustment plots, make sure you run the tool and supply the correct directory to asp_plot.\n\n"
            )

    # Compile report
    processing_parameters = ProcessingParameters(
        processing_directory=directory,
        bundle_adjust_directory=bundle_adjust_directory,
        stereo_directory=stereo_directory,
    )
    processing_parameters_dict = processing_parameters.from_log_files()

    compile_report(
        plots_directory,
        processing_parameters_dict,
        report_pdf_path,
        report_title=report_title,
    )

    shutil.rmtree(plots_directory)

    print(f"\n\nReport saved to {report_pdf_path}\n\n")


if __name__ == "__main__":
    main()
