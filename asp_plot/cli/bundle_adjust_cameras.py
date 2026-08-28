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
    "--map-crs",
    prompt=False,
    default=None,
    help="CRS for the camera-center geometry, as EPSG:XXXX (e.g. the site's UTM "
    "zone). Only affects the returned geometry; the east/north/up offsets do "
    "not depend on it. Default: geographic coordinates (EPSG:4326).",
)
@click.option(
    "--original-cameras-directory",
    prompt=False,
    default=None,
    help="Directory holding the original .xml cameras, used only for DigitalGlobe "
    "runs that lack *.adjusted_state.json. If not supplied, the bundle_adjust "
    "directory and its parent are searched automatically.",
)
@click.option(
    "--title",
    prompt=False,
    default=None,
    help="Optional title for the summary figure. Default: None.",
)
@click.option(
    "--output-directory",
    prompt=False,
    default=None,
    help="Directory to save the figure. Default: the bundle_adjust directory itself.",
)
@click.option(
    "--output-filename",
    prompt=False,
    default="bundle_adjust_cameras_summary.png",
    help="Figure filename. Default: bundle_adjust_cameras_summary.png.",
)
def main(
    directory,
    map_crs,
    original_cameras_directory,
    title,
    output_directory,
    output_filename,
):
    """
    Visualize before/after camera positions from a bundle_adjust folder.

    Reads the self-contained camera products written by bundle_adjust
    (``*.adjust``, ``*.adjusted_state.json``, and, when present,
    ``*camera_offsets.txt``) and produces a summary: per-camera bars of the
    horizontal and vertical camera-center change, above a per-camera satellite
    cartoon of the roll/pitch/yaw orientation change (labeled with the actual
    degrees changed).

    Unlike ``csm_camera_plot``, this does not require the pre-adjustment
    original camera files -- it works directly on the bundle_adjust output.
    """
    # The reader takes a root + subdirectory (matching ReadBundleAdjustFiles), but
    # this tool is self-contained on the BA folder, so accept a single path and
    # split it into (parent, basename) internally.
    parent, ba_dir = os.path.split(directory.rstrip("/\\"))
    reader = ReadBundleAdjustCameras(parent, ba_dir)
    gdf = reader.get_camera_optimization_gdf(
        map_crs=int(map_crs.split(":")[-1]) if map_crs else None,
        original_cameras_directory=original_cameras_directory,
    )
    # Default to saving in the bundle_adjust directory so a bare CLI call always
    # writes a figure somewhere sensible (the command does not display a window).
    if output_directory is None:
        output_directory = reader.full_directory
    PlotBundleAdjustCameras(gdf, title=title).summary_plot(
        save_dir=output_directory, fig_fn=output_filename
    )


if __name__ == "__main__":
    main()
