import os
import shutil
from datetime import datetime, timezone
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
    help="If True, add a basemaps to the figures, which requires internet connection. Default: True.",
)
@click.option(
    "--plot_icesat",
    prompt=False,
    default=True,
    help="If True, plot an ICESat-2 difference plot with the DEM result. This requires internet connection to request ICESat data. Default: True.",
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
    dem_filename,
    dem_gsd,
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

    if report_title is None:
        report_title = os.path.split(directory.rstrip("/\\"))[-1]

    if report_filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_filename = f"asp_plot_report_{report_title}_{timestamp}.pdf"

    # Save output report in the same directory as processed DEM
    report_pdf_path = os.path.join(
        directory, os.path.join(stereo_directory, report_filename)
    )

    figure_counter = count(0)

    # TODO: map crs should be set by output DEM.tif or default to a local orthographic projection, not EPSG:4326
    #  https://github.com/uw-cryo/asp_plot/issues/76
    if map_crs is None:
        map_crs = "EPSG:4326"
        print(
            f"\nNo map projection supplied. Default is {map_crs}. If you want a different projection, use the --map_crs flag.\n"
        )
        add_basemap = False

    # TODO: Centralize this in plotting utils, should not need ctx import in the CLI wrapper
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
        dem_fn=dem_filename,
        dem_gsd=dem_gsd,
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
    plotter = SceneGeometryPlotter(directory, add_basemap=add_basemap)
    plotter.dg_geom_plot(
        save_dir=plots_directory, fig_fn=f"{next(figure_counter):02}.png"
    )

    # ICESat-2 comparison
    if plot_icesat:
        icesat = Altimetry(directory=directory, dem_fn=asp_dem)

        icesat.request_atl06sr_multi_processing(
            processing_levels=["all", "ground"],
            save_to_parquet=False,
        )

        icesat.filter_esa_worldcover(filter_out="water")

        icesat.predefined_temporal_filter_atl06sr()

        icesat.mapview_plot_atl06sr_to_dem(
            key="all",
            save_dir=plots_directory,
            fig_fn=f"{next(figure_counter):02}.png",
            **ctx_kwargs,
        )

        icesat.histogram(
            key="all",
            plot_aligned=False,
            save_dir=plots_directory,
            fig_fn=f"{next(figure_counter):02}.png",
        )

        icesat.mapview_plot_atl06sr_to_dem(
            key="ground_seasonal",
            save_dir=plots_directory,
            fig_fn=f"{next(figure_counter):02}.png",
            **ctx_kwargs,
        )

        icesat.histogram(
            key="ground_seasonal",
            plot_aligned=False,
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
