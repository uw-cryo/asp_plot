import os

import click

from asp_plot.bundle_adjust import PlotBundleAdjustCameras, ReadBundleAdjustCameras


@click.command()
@click.option(
    "--directory",
    prompt=True,
    default="",
    help="Path to the bundle_adjust output directory (the folder holding the "
    "*.adjust, *.adjusted_state.json, and optional *camera_offsets.txt files). "
    "No default. Must be supplied.",
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
    help="Directory to save the figure. Default: the bundle_adjust directory itself.",
)
@click.option(
    "--fig_fn",
    prompt=False,
    default="bundle_adjust_cameras_summary.png",
    help="Figure filename. Default: bundle_adjust_cameras_summary.png.",
)
def main(directory, map_crs, title, save_dir, fig_fn):
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
    # The reader takes a root + subdirectory (matching ReadBundleAdjustFiles), but
    # this tool is self-contained on the BA folder, so accept a single path and
    # split it into (parent, basename) internally.
    parent, ba_dir = os.path.split(directory.rstrip(os.sep))
    reader = ReadBundleAdjustCameras(parent, ba_dir)
    gdf = reader.get_camera_optimization_gdf(
        map_crs=int(map_crs.split(":")[-1]) if map_crs else None
    )
    # Default to saving in the bundle_adjust directory so a bare CLI call always
    # writes a figure somewhere sensible (the command does not display a window).
    if save_dir is None:
        save_dir = reader.full_directory
    PlotBundleAdjustCameras(gdf, title=title).summary_plot(
        save_dir=save_dir, fig_fn=fig_fn
    )


if __name__ == "__main__":
    main()
