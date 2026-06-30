import os

import click

from asp_plot.stereo_geometry import StereoGeometryPlotter


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path())
@click.option(
    "--directory",
    default=None,
    help=(
        "Directory containing XML files for stereo geometry analysis. "
        "Used when no positional INPUTS are given. Default: current directory."
    ),
)
@click.option(
    "--add_basemap",
    prompt=False,
    default=True,
    help="If True, add a basemap to the figures, which requires internet connection. Default: True.",
)
@click.option(
    "--output_directory",
    prompt=False,
    default=None,
    help="Directory to save the output plot. Default: Input directory.",
)
@click.option(
    "--output_filename",
    prompt=False,
    default=None,
    help=(
        "Filename for the output plot. Default: Directory name with "
        "_stereo_geom.png suffix. With more than two scenes this is the stem for "
        "the per-pair and overview figures."
    ),
)
def main(
    inputs,
    directory,
    add_basemap,
    output_directory,
    output_filename,
):
    """
    Generate stereo geometry plots for WorldView XML files.

    This tool creates a skyplot and map visualization of the satellite positions
    and ground footprints. INPUTS may be any mix of XML files, directories, and
    glob patterns and need not follow a fixed directory structure, e.g.:

        stereo_geom *.XML

        stereo_geom scene1.xml scene2.xml

        stereo_geom my_delivery_dir/

    If no INPUTS are given, --directory is used (default: current directory).
    """
    # Positional INPUTS take precedence; otherwise fall back to --directory.
    if inputs:
        inputs = [os.path.expanduser(i) for i in inputs]
        plotter = StereoGeometryPlotter(inputs=inputs, add_basemap=add_basemap)
        source_desc = " ".join(inputs)
        # Base directory for default output paths (resolved by the parser).
        base_directory = plotter.directory
    else:
        base_directory = os.path.expanduser(directory or "./")
        plotter = StereoGeometryPlotter(
            directory=base_directory, add_basemap=add_basemap
        )
        source_desc = base_directory

    print(f"\nProcessing stereo geometry for XML files in {source_desc}\n")

    # Derive default output directory/filename from the base directory.
    dir_name = os.path.split(base_directory.rstrip("/\\"))[-1]
    if output_directory is None:
        output_directory = base_directory
    else:
        output_directory = os.path.expanduser(output_directory)
    if output_filename is None:
        output_filename = f"{dir_name}_stereo_geom.png"

    os.makedirs(output_directory, exist_ok=True)

    saved = plotter.dg_geom_plot(save_dir=output_directory, fig_fn=output_filename)

    if saved:
        if len(saved) == 1:
            print(
                f"\nStereo geometry plot saved to {os.path.join(output_directory, saved[0])}\n"
            )
        else:
            print(f"\nStereo geometry plots saved ({len(saved)}):")
            for fn in saved:
                print(f"  {os.path.join(output_directory, fn)}")
            print()


if __name__ == "__main__":
    main()
