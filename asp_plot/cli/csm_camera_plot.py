import click
import contextily as ctx

from asp_plot.csm_camera import csm_camera_summary_plot


@click.command()
@click.option(
    "--original-cameras",
    prompt=True,
    default="",
    help="Original camera files, supplied as comma separated list 'path/to/original_camera_1,path/to/original_camera_2'. No default. Must be supplied.",
)
@click.option(
    "--optimized-cameras",
    prompt=True,
    default="",
    help="Optimized camera files, supplied as comma separated list 'path/to/optimized_camera_1,path/to/optimized_camera_2'. No default. Must be supplied.",
)
@click.option(
    "--map-crs",
    prompt=False,
    default=None,
    help="UTM EPSG code for map projection. As EPSG:XXXX. If not supplied, the map will be plotted in original camera coordinates of EPSG:4978 (ECEF).",
)
@click.option(
    "--title",
    prompt=False,
    default=None,
    help="Optional short title to append to figure output. Default: None.",
)
@click.option(
    "--no-trim",
    is_flag=True,
    default=False,
    help="Do not trim the plotted positions to the first and last camera image lines (trimmed by default).",
)
@click.option(
    "--shared-scales",
    is_flag=True,
    default=False,
    help="Share the position and angle difference scales between the cameras.",
)
@click.option(
    "--log-scale-positions",
    is_flag=True,
    default=False,
    help="Log-scale the position difference plots.",
)
@click.option(
    "--log-scale-angles",
    is_flag=True,
    default=False,
    help="Log-scale the angle difference plots.",
)
@click.option(
    "--upper-magnitude-percentile",
    prompt=False,
    default=95,
    help="Percentile to use for the upper limit of the mapview colorbars. Default: 95.",
)
@click.option(
    "--figure-size",
    prompt=False,
    default="20,15",
    help="Figure size as width,height. Default: 20,15.",
    callback=lambda ctx, param, value: tuple(map(int, value.split(","))),
)
@click.option(
    "--output-directory",
    prompt=False,
    default=None,
    help="Directory to save the figure. Default: None, which does not save the figure.",
)
@click.option(
    "--output-filename",
    prompt=False,
    default="csm_camera_summary_plot.png",
    help="Figure filename. Default: csm_camera_summary_plot.png.",
)
@click.option(
    "--add-basemap",
    is_flag=True,
    default=False,
    help="Add a contextily basemap to the figure, which requires an internet connection.",
)
def main(
    original_cameras,
    optimized_cameras,
    map_crs,
    title,
    no_trim,
    shared_scales,
    log_scale_positions,
    log_scale_angles,
    upper_magnitude_percentile,
    figure_size,
    output_directory,
    output_filename,
    add_basemap,
):
    """
    Create diagnostic plots for CSM camera model adjustments.

    Analyzes the changes between original and optimized camera models after bundle
    adjustment or jitter correction. Generates plots showing position and angle differences
    along the satellite trajectory, as well as a mapview of the camera footprints.
    """
    original_cameras = original_cameras.split(",")
    optimized_cameras = optimized_cameras.split(",")

    cam1_list = [original_cameras[0], optimized_cameras[0]]
    if len(original_cameras) > 1 and len(optimized_cameras) > 1:
        cam2_list = [original_cameras[1], optimized_cameras[1]]
    else:
        cam2_list = None

    ctx_kwargs = {
        "crs": map_crs,
        "source": ctx.providers.Esri.WorldImagery,
        "attribution_size": 0,
        "alpha": 0.5,
    }

    csm_camera_summary_plot(
        cam1_list,
        cam2_list,
        map_crs.split(":")[-1] if map_crs else None,
        title=title,
        trim=not no_trim,
        shared_scales=shared_scales,
        log_scale_positions=log_scale_positions,
        log_scale_angles=log_scale_angles,
        upper_magnitude_percentile=upper_magnitude_percentile,
        figsize=figure_size,
        save_dir=output_directory,
        fig_fn=output_filename,
        add_basemap=add_basemap,
        **ctx_kwargs,
    )


if __name__ == "__main__":
    main()
