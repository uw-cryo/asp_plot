import click
import contextily as ctx

from asp_plot.csm_camera import csm_camera_summary_plot


@click.command()
@click.option(
    "--original_cameras",
    prompt=True,
    default="",
    help="Original camera files, supplied as comma separated list 'path/to/original_camera_1,path/to/original_camera_2'. No default. Must be supplied.",
)
@click.option(
    "--optimized_cameras",
    prompt=True,
    default="",
    help="Optimized camera files, supplied as comma separated list 'path/to/optimized_camera_1,path/to/optimized_camera_2'. No default. Must be supplied.",
)
@click.option(
    "--map_crs",
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
    "--trim",
    prompt=False,
    default=True,
    help="Trim the beginning and end of the positions plotted to the first and last camera image lines. Default: True.",
)
@click.option(
    "--shared_scales",
    prompt=False,
    default=False,
    help="If True, the position and angle difference scales are shared between for each camera. Default: False.",
)
@click.option(
    "--log_scale_positions",
    prompt=False,
    default=False,
    help="If True, the position difference scales are log scaled. Default: False.",
)
@click.option(
    "--log_scale_angles",
    prompt=False,
    default=False,
    help="If True, the angle difference scales are log scaled. Default: False.",
)
@click.option(
    "--upper_magnitude_percentile",
    prompt=False,
    default=95,
    help="Percentile to use for the upper limit of the mapview colorbars. Default: 95.",
)
@click.option(
    "--figsize",
    prompt=False,
    default="20,15",
    help="Figure size as width,height. Default: 20,15.",
    callback=lambda ctx, param, value: tuple(map(int, value.split(","))),
)
@click.option(
    "--save_dir",
    prompt=False,
    default=None,
    help="Directory to save the figure. Default: None, which does not save the figure.",
)
@click.option(
    "--fig_fn",
    prompt=False,
    default="csm_camera_summary_plot.png",
    help="Figure filename. Default: csm_camera_summary_plot.png.",
)
@click.option(
    "--add_basemap",
    prompt=False,
    default=False,
    help="If True, add a contextily basemap to the figure, which requires internet connection. Default: False.",
)
def main(
    original_cameras,
    optimized_cameras,
    map_crs,
    title,
    trim,
    shared_scales,
    log_scale_positions,
    log_scale_angles,
    upper_magnitude_percentile,
    figsize,
    save_dir,
    fig_fn,
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
        map_crs,
        title=title,
        trim=trim,
        shared_scales=shared_scales,
        log_scale_positions=log_scale_positions,
        log_scale_angles=log_scale_angles,
        upper_magnitude_percentile=upper_magnitude_percentile,
        figsize=figsize,
        save_dir=save_dir,
        fig_fn=fig_fn,
        add_basemap=add_basemap,
        **ctx_kwargs,
    )


if __name__ == "__main__":
    main()
