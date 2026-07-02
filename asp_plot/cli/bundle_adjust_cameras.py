import click

from asp_plot.bundle_adjust import PlotBundleAdjustCameras, ReadBundleAdjustCameras


@click.command()
@click.option(
    "--directory",
    prompt=True,
    default="",
    help="Root ASP processing directory. No default. Must be supplied.",
)
@click.option(
    "--bundle_adjust_directory",
    prompt=True,
    default="ba",
    help="Subdirectory (relative to --directory) holding the bundle_adjust output. Default: ba.",
)
@click.option(
    "--map_crs",
    prompt=False,
    default=None,
    help="UTM EPSG code for the map projection, as EPSG:XXXX. Recommended so the "
    "position-change quiver arrows align with the map axes. If not supplied, "
    "geometry is returned in geographic coordinates (EPSG:4326).",
)
@click.option(
    "--title",
    prompt=False,
    default=None,
    help="Optional title for the summary figure. Default: None.",
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
    default="bundle_adjust_cameras_summary.png",
    help="Figure filename. Default: bundle_adjust_cameras_summary.png.",
)
def main(directory, bundle_adjust_directory, map_crs, title, save_dir, fig_fn):
    """
    Visualize before/after camera positions from a bundle_adjust folder.

    Reads the self-contained camera products written by bundle_adjust
    (``*.adjust``, ``*.adjusted_state.json``, and, when present,
    ``*camera_offsets.txt``) and produces a three-panel summary: a map-view
    quiver of the horizontal camera-center shift (vertical change as color),
    per-camera center-displacement bars, and an orientation-change quiver.

    Unlike ``csm_camera_plot``, this does not require the pre-adjustment
    original camera files -- it works directly on the bundle_adjust output.
    """
    reader = ReadBundleAdjustCameras(directory, bundle_adjust_directory)
    gdf = reader.get_camera_optimization_gdf(
        map_crs=int(map_crs.split(":")[-1]) if map_crs else None
    )
    PlotBundleAdjustCameras(gdf, title=title).summary_plot(
        save_dir=save_dir, fig_fn=fig_fn
    )


if __name__ == "__main__":
    main()
