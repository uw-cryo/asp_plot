import os
import shutil
from datetime import datetime, timezone
from itertools import count

import click
import contextily as ctx

from asp_plot.altimetry import Altimetry
from asp_plot.bundle_adjust import PlotBundleAdjustFiles, ReadBundleAdjustFiles
from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.report import ReportMetadata, ReportSection, compile_report
from asp_plot.scenes import ScenePlotter
from asp_plot.stereo import StereoPlotter
from asp_plot.stereo_geometry import StereoGeometryPlotter
from asp_plot.utils import Raster


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
    help="Projection for ICESat and bundle adjustment plots. As EPSG:XXXX. Default: None, which will use the projection of the ASP DEM, and fall back on EPSG:4326 if not found.",
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
    "--icesat_filter_date",
    prompt=False,
    default=None,
    help="The central date (YYYY-MM-DD) to temporally filter ICESat-2 ATL06 data around. This should be the capture date of the scenes. Default: None, which attempts to read scene capture date from metadata (currently only works for WorldView scenes).",
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
    icesat_filter_date,
    plot_geometry,
    subset_km,
    report_filename,
    report_title,
):
    """
    Generate a comprehensive report of ASP processing results.

    Creates a series of diagnostic plots for stereo processing, bundle adjustment,
    ICESat-2 comparisons, and more. All plots are combined into a single PDF report
    with processing parameters and summary information.
    """
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

    sections = []
    figure_counter = count(0)

    # Initialize StereoPlotter early (needed for DEM info and multiple plot types)
    stereo_plotter = StereoPlotter(
        directory,
        stereo_directory,
        reference_dem=reference_dem,
        dem_fn=dem_filename,
        dem_gsd=dem_gsd,
    )
    asp_dem = stereo_plotter.dem_fn

    # Set map CRS from output DEM and collect DEM metadata
    report_metadata = None
    if map_crs is None:
        if asp_dem and os.path.exists(asp_dem):
            try:
                dem_raster = Raster(asp_dem)
                epsg_code = dem_raster.get_epsg_code()
                map_crs = f"EPSG:{epsg_code}"
                print(f"\nUsing map projection from DEM: {map_crs}\n")
            except Exception as e:
                print(
                    f"\nError getting projection from DEM: {e}. Using default projection EPSG:4326. If you want a different projection, use the --map_crs flag.\n"
                )
                map_crs = "EPSG:4326"

    # Collect DEM metadata for the report title page
    if asp_dem and os.path.exists(asp_dem):
        try:
            dem_raster = Raster(asp_dem)
            dem_data = dem_raster.read_array()
            total_pixels = dem_data.size
            nodata_pixels = dem_data.mask.sum() if hasattr(dem_data.mask, "sum") else 0
            nodata_pct = (nodata_pixels / total_pixels * 100) if total_pixels else 0.0
            valid = dem_data.compressed()
            elev_range = (
                (float(valid.min()), float(valid.max())) if valid.size else (0, 0)
            )
            report_metadata = ReportMetadata(
                dem_dimensions=(dem_raster.ds.width, dem_raster.ds.height),
                dem_gsd_m=dem_raster.get_gsd(),
                dem_crs=map_crs or "",
                dem_nodata_percent=nodata_pct,
                dem_elevation_range=elev_range,
                dem_filename=os.path.basename(asp_dem),
                reference_dem=reference_dem or "",
            )
        except Exception as e:
            print(f"\nCould not collect DEM metadata: {e}\n")

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

    # ---- Section 1: Input Scenes ----
    fig_fn = f"{next(figure_counter):02}.png"
    scene_plotter = ScenePlotter(directory, stereo_directory, title="Input Scenes")
    scene_plotter.plot_scenes(save_dir=plots_directory, fig_fn=fig_fn)
    sections.append(
        ReportSection(
            title="Input Scenes",
            image_path=os.path.join(plots_directory, fig_fn),
            caption="Left and right input scenes used for stereo processing.",
        )
    )

    # ---- Section 2: Stereo Geometry (conditional) ----
    if plot_geometry:
        fig_fn = f"{next(figure_counter):02}.png"
        geom_plotter = StereoGeometryPlotter(directory, add_basemap=add_basemap)
        geom_plotter.dg_geom_plot(save_dir=plots_directory, fig_fn=fig_fn)
        sections.append(
            ReportSection(
                title="Stereo Geometry",
                image_path=os.path.join(plots_directory, fig_fn),
                caption="Stereo acquisition geometry skyplot and map view showing satellite viewing angles and scene footprints.",
            )
        )

    # ---- Section 3: Match Points ----
    fig_fn = f"{next(figure_counter):02}.png"
    stereo_plotter.title = "Stereo Match Points"
    stereo_plotter.plot_match_points(save_dir=plots_directory, fig_fn=fig_fn)
    sections.append(
        ReportSection(
            title="Match Points",
            image_path=os.path.join(plots_directory, fig_fn),
            caption="Interest point matches between left and right images identified during stereo correlation.",
        )
    )

    # ---- Section 4: Detailed Hillshade ----
    fig_fn = f"{next(figure_counter):02}.png"
    stereo_plotter.title = "Hillshade with details"
    stereo_plotter.plot_detailed_hillshade(
        subset_km=subset_km, save_dir=plots_directory, fig_fn=fig_fn
    )
    sections.append(
        ReportSection(
            title="Detailed Hillshade",
            image_path=os.path.join(plots_directory, fig_fn),
            caption=f"DEM hillshade with {subset_km} km detail subset in second row. If available, corresponding mapprojected ortho image subsets are displayed in the bottom row.",
        )
    )

    # ---- Section 5: DEM Results ----
    fig_fn = f"{next(figure_counter):02}.png"
    stereo_plotter.title = "Stereo DEM Results"
    stereo_plotter.plot_dem_results(save_dir=plots_directory, fig_fn=fig_fn)
    sections.append(
        ReportSection(
            title="DEM Results",
            image_path=os.path.join(plots_directory, fig_fn),
            caption="Output DEM with intersection error map and difference relative to the reference DEM used in processing.",
        )
    )

    # ---- Section 6: Disparity ----
    fig_fn = f"{next(figure_counter):02}.png"
    stereo_plotter.title = "Disparity (pixels)"
    stereo_plotter.plot_disparity(
        unit="pixels", quiver=True, save_dir=plots_directory, fig_fn=fig_fn
    )
    sections.append(
        ReportSection(
            title="Disparity",
            image_path=os.path.join(plots_directory, fig_fn),
            caption="Horizontal and vertical disparity maps in pixels with quiver overlay.",
        )
    )

    # ---- Sections 7-10: ICESat-2 (conditional) ----
    if plot_icesat:
        icesat = Altimetry(directory=directory, dem_fn=asp_dem)

        icesat.request_atl06sr_multi_processing(
            processing_levels=["all", "ground"],
            save_to_parquet=True,
        )

        icesat.filter_esa_worldcover(filter_out="water")

        if icesat_filter_date is not None:
            icesat_filter_date = datetime.strptime(
                icesat_filter_date, "%Y-%m-%d"
            ).date()

        icesat.predefined_temporal_filter_atl06sr(date=icesat_filter_date)

        fig_fn = f"{next(figure_counter):02}.png"
        icesat.mapview_plot_atl06sr_to_dem(
            key="all",
            save_dir=plots_directory,
            fig_fn=fig_fn,
            map_crs=map_crs,
            **ctx_kwargs,
        )
        sections.append(
            ReportSection(
                title="ICESat-2 ATL06-SR Map (All)",
                image_path=os.path.join(plots_directory, fig_fn),
                caption="ICESat-2 ATL06-SR elevation differences (all processing levels) vs. ASP DEM.",
            )
        )

        fig_fn = f"{next(figure_counter):02}.png"
        icesat.histogram(
            key="all",
            plot_aligned=False,
            save_dir=plots_directory,
            fig_fn=fig_fn,
        )
        sections.append(
            ReportSection(
                title="ICESat-2 ATL06-SR Histogram (All)",
                image_path=os.path.join(plots_directory, fig_fn),
                caption="Distribution of elevation differences between ICESat-2 ATL06-SR (all) and ASP DEM.",
            )
        )

        fig_fn = f"{next(figure_counter):02}.png"
        icesat.mapview_plot_atl06sr_to_dem(
            key="ground_seasonal",
            save_dir=plots_directory,
            fig_fn=fig_fn,
            map_crs=map_crs,
            **ctx_kwargs,
        )
        sections.append(
            ReportSection(
                title="ICESat-2 ATL06-SR Map (Ground, Seasonal)",
                image_path=os.path.join(plots_directory, fig_fn),
                caption="ICESat-2 ATL06-SR elevation differences (ground, seasonally filtered) vs. ASP DEM.",
            )
        )

        fig_fn = f"{next(figure_counter):02}.png"
        icesat.histogram(
            key="ground_seasonal",
            plot_aligned=False,
            save_dir=plots_directory,
            fig_fn=fig_fn,
        )
        sections.append(
            ReportSection(
                title="ICESat-2 ATL06-SR Histogram (Ground, Seasonal)",
                image_path=os.path.join(plots_directory, fig_fn),
                caption="Distribution of elevation differences between ICESat-2 ATL06-SR (ground, seasonal) and ASP DEM.",
            )
        )

    # ---- Sections 11+: Bundle Adjustment (conditional) ----
    if bundle_adjust_directory:
        try:
            ba_files = ReadBundleAdjustFiles(directory, bundle_adjust_directory)
            resid_initial_gdf, resid_final_gdf = (
                ba_files.get_initial_final_residuals_gdfs()
            )

            plotter = PlotBundleAdjustFiles(
                [resid_initial_gdf, resid_final_gdf],
                lognorm=True,
                title="Bundle Adjust Initial and Final Residuals (Log Scale)",
            )

            fig_fn = f"{next(figure_counter):02}.png"
            plotter.plot_n_gdfs(
                column_name="mean_residual",
                cbar_label="Mean residual (px)",
                map_crs=map_crs,
                save_dir=plots_directory,
                fig_fn=fig_fn,
                **ctx_kwargs,
            )
            sections.append(
                ReportSection(
                    title="Bundle Adjust Residuals (Log Scale)",
                    image_path=os.path.join(plots_directory, fig_fn),
                    caption="Initial and final bundle adjustment residuals on a logarithmic scale.",
                )
            )

            plotter.lognorm = False
            plotter.title = "Bundle Adjust Initial and Final Residuals (Linear Scale)"

            fig_fn = f"{next(figure_counter):02}.png"
            plotter.plot_n_gdfs(
                column_name="mean_residual",
                cbar_label="Mean residual (px)",
                common_clim=False,
                map_crs=map_crs,
                save_dir=plots_directory,
                fig_fn=fig_fn,
                **ctx_kwargs,
            )
            sections.append(
                ReportSection(
                    title="Bundle Adjust Residuals (Linear Scale)",
                    image_path=os.path.join(plots_directory, fig_fn),
                    caption="Initial and final bundle adjustment residuals on a linear scale.",
                )
            )

            # Map-projected residuals (requires reference DEM in bundle_adjust)
            try:
                resid_mapprojected_gdf = ba_files.get_mapproj_residuals_gdf()

                plotter = PlotBundleAdjustFiles(
                    [resid_mapprojected_gdf],
                    title="Bundle Adjust Midpoint distance between\nfinal interest points projected onto reference DEM",
                )

                fig_fn = f"{next(figure_counter):02}.png"
                plotter.plot_n_gdfs(
                    column_name="mapproj_ip_dist_meters",
                    cbar_label="Interest point distance (m)",
                    map_crs=map_crs,
                    save_dir=plots_directory,
                    fig_fn=fig_fn,
                    **ctx_kwargs,
                )
                sections.append(
                    ReportSection(
                        title="Map-Projected Residuals",
                        image_path=os.path.join(plots_directory, fig_fn),
                        caption="Midpoint distance between final interest points projected onto the reference DEM used in processing.",
                    )
                )
            except ValueError as e:
                print(f"\n\nSkipping map-projected residuals plot: {e}\n\n")

            # Geodiff plots (requires reference DEM in bundle_adjust with --mapproj-dem flag)
            try:
                geodiff_initial_gdf, geodiff_final_gdf = (
                    ba_files.get_initial_final_geodiff_gdfs()
                )

                plotter = PlotBundleAdjustFiles(
                    [geodiff_initial_gdf, geodiff_final_gdf],
                    lognorm=False,
                    title="Bundle Adjust Initial and Final Geodiff vs. Reference DEM",
                )

                fig_fn = f"{next(figure_counter):02}.png"
                plotter.plot_n_gdfs(
                    column_name="height_diff_meters",
                    cbar_label="Height difference (m)",
                    map_crs=map_crs,
                    cmap="RdBu",
                    symm_clim=True,
                    save_dir=plots_directory,
                    fig_fn=fig_fn,
                    **ctx_kwargs,
                )
                sections.append(
                    ReportSection(
                        title="Geodiff vs. Reference DEM",
                        image_path=os.path.join(plots_directory, fig_fn),
                        caption="Initial and final geodiff height differences compared to the reference DEM used in processing.",
                    )
                )
            except ValueError as e:
                print(
                    f"\n\nSkipping geodiff plots (requires --mapproj-dem flag in bundle_adjust): {e}\n\n"
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
        sections,
        processing_parameters_dict,
        report_pdf_path,
        report_title=report_title,
        report_metadata=report_metadata,
    )

    shutil.rmtree(plots_directory)

    print(f"\n\nReport saved to {report_pdf_path}\n\n")


if __name__ == "__main__":
    main()
