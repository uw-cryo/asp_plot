import os

import click

from asp_plot.stereo_geometry import StereoGeometryPlotter


@click.command()
@click.argument("inputs", nargs=-1, type=click.Path())
@click.option(
    "--directory",
    default=None,
    help=(
        "Directory containing camera metadata files (XMLs, or images carrying "
        "RPCs) for stereo geometry analysis. "
        "Used when no positional INPUTS are given. Default: current directory."
    ),
)
@click.option(
    "--no-basemap",
    is_flag=True,
    default=False,
    help="Skip the figure basemaps (basemaps are added by default, which requires an internet connection).",
)
@click.option(
    "--output-directory",
    prompt=False,
    default=None,
    help="Directory to save the output plot. Default: Input directory.",
)
@click.option(
    "--output-filename",
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
    no_basemap,
    output_directory,
    output_filename,
):
    """
    Generate stereo geometry plots from satellite camera metadata files.

    The sensor is detected from the files themselves: WorldView (and other
    DigitalGlobe-heritage) XML, Airbus DIMAP v2 (Pléiades 1A/1B and Neo,
    SPOT 6/7, PeruSat-1), DIMAP v1 (SPOT 5, ALOS PRISM), ASTER
    (ASP gen_aster camera XML), and RPC-only products such as Cartosat-1 and
    Deimos, whose camera model lives in the image itself.

    This tool creates a skyplot and map visualization of the satellite positions
    and ground footprints. INPUTS may be any mix of camera metadata files,
    images carrying RPCs, directories, and glob patterns, and need not follow a
    fixed directory structure, e.g.:

        stereo_geom *.XML

        stereo_geom scene1.xml scene2.xml

        stereo_geom fore.tif aft.tif

        stereo_geom my_delivery_dir/

    If no INPUTS are given, --directory is used (default: current directory).
    """
    # Positional INPUTS take precedence; otherwise fall back to --directory.
    if inputs:
        inputs = [os.path.expanduser(i) for i in inputs]
        plotter = StereoGeometryPlotter(inputs=inputs, add_basemap=not no_basemap)
        source_desc = " ".join(inputs)
        # Base directory for default output paths (resolved by the parser).
        base_directory = plotter.directory
    else:
        base_directory = os.path.expanduser(directory or "./")
        plotter = StereoGeometryPlotter(
            directory=base_directory, add_basemap=not no_basemap
        )
        source_desc = base_directory

    print(f"\nProcessing stereo geometry for camera metadata in {source_desc}\n")

    # Derive default output directory/filename from the base directory.
    dir_name = os.path.split(base_directory.rstrip("/\\"))[-1]
    if output_directory is None:
        output_directory = base_directory
    else:
        output_directory = os.path.expanduser(output_directory)
    if output_filename is None:
        output_filename = f"{dir_name}_stereo_geom.png"

    os.makedirs(output_directory, exist_ok=True)

    saved = plotter.stereo_geom_plot(save_dir=output_directory, fig_fn=output_filename)

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
