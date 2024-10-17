import click
import contextily as ctx

from asp_plot.camera_optimization import summary_plot_two_camera_optimization


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
    help="UTM EPSG code for map projection. If not supplied, the map will be plotted in original camera coordinates of EPSG:4978 (ECEF).",
)
@click.option(
    "--title",
    prompt=False,
    default=None,
    help="Optional short title to append to figure output. Default: None",
)
@click.option(
    "--trim",
    prompt=False,
    default=False,
    help="Trim the beginning and end of the geodataframes. Default: False",
)
@click.option(
    "--near_zero_tolerance",
    prompt=False,
    default=1e-3,
    help="If trim is True, the tolerance for near zero values of the camera position differences to trim from the beginning and end. Default: 1e-3",
)
@click.option(
    "--trim_percentage",
    prompt=False,
    default=5,
    help="If trim is ture, the extra percentage of the camera positions to trim from the beginning and end. Default: 5",
)
@click.option(
    "--shared_scales",
    prompt=False,
    default=False,
    help="If True, the position and angle difference scales are shared between for each camera. Default: False",
)
@click.option(
    "--log_scale_positions",
    prompt=False,
    default=False,
    help="If True, the position difference scales are log scaled. Default: False",
)
@click.option(
    "--log_scale_angles",
    prompt=False,
    default=False,
    help="If True, the angle difference scales are log scaled. Default: False",
)
@click.option(
    "--upper_magnitude_percentile",
    prompt=False,
    default=95,
    help="Percentile to use for the upper limit of the mapview colorbars. Default: 95",
)
@click.option(
    "--figsize",
    prompt=False,
    default="20,15",
    help="Figure size as width,height. Default: 20,15",
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
    default="camera_optimization_summary_plot.png",
    help="Figure filename. Default: camera_optimization_summary_plot.png.",
)
@click.option(
    "--add_basemap",
    prompt=False,
    default=False,
    help="If True, add a contextily basemap to the figure, which requires internet connection. Default: False",
)
def main(
    original_cameras,
    optimized_cameras,
    map_crs,
    title,
    trim,
    trim_percentage,
    near_zero_tolerance,
    shared_scales,
    log_scale_positions,
    log_scale_angles,
    upper_magnitude_percentile,
    figsize,
    save_dir,
    fig_fn,
    add_basemap,
):
    original_cameras = original_cameras.split(",")
    optimized_cameras = optimized_cameras.split(",")

    cam1_list = [original_cameras[0], optimized_cameras[0]]
    cam2_list = [original_cameras[1], optimized_cameras[1]]

    ctx_kwargs = {
        "crs": f"EPSG:{map_crs}",
        "source": ctx.providers.Esri.WorldImagery,
        "attribution_size": 0,
        "alpha": 0.5,
    }

    summary_plot_two_camera_optimization(
        cam1_list,
        cam2_list,
        map_crs,
        title=title,
        trim=trim,
        trim_percentage=trim_percentage,
        near_zero_tolerance=near_zero_tolerance,
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
