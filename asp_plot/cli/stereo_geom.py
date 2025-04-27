import os

import click

from asp_plot.stereo_geometry import StereoGeometryPlotter


@click.command()
@click.option(
    "--directory",
    prompt=True,
    default="./",
    help="Directory containing XML files for stereo geometry analysis. Default: current directory.",
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
    help="Filename for the output plot. Default: Directory name with _stereo_geom.png suffix.",
)
def main(
    directory,
    add_basemap,
    output_directory,
    output_filename,
):
    """
    Generate stereo geometry plots for DigitalGlobe/Maxar XML files.
    This tool creates a skyplot and map visualization of the satellite positions and ground footprints.
    """
    print(f"\nProcessing stereo geometry for XML files in {directory}\n")

    # Get directory name for default output filename
    dir_name = os.path.split(directory.rstrip("/\\"))[-1]

    # Set default output directory to input directory
    if output_directory is None:
        output_directory = directory

    # Set default output filename using directory name
    if output_filename is None:
        output_filename = f"{dir_name}_stereo_geom.png"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Create the geometry plotter and generate the plot
    plotter = StereoGeometryPlotter(directory, add_basemap=add_basemap)
    plotter.dg_geom_plot(save_dir=output_directory, fig_fn=output_filename)

    print(
        f"\nStereo geometry plot saved to {os.path.join(output_directory, output_filename)}\n"
    )


if __name__ == "__main__":
    main()
