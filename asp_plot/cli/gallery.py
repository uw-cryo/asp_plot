import os

import click

from asp_plot.gallery import GalleryPlotter


@click.command()
@click.option(
    "--directory",
    prompt=False,
    default="./",
    help="Directory to search for rasters. Default: current directory.",
)
@click.option(
    "--pattern",
    prompt=False,
    default="*-DEM.tif",
    help="Glob pattern for rasters within the directory. Default: '*-DEM.tif'.",
)
@click.argument("files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--hillshade/--no-hillshade",
    default=True,
    help="Draw a gray hillshade underlay beneath each DEM. Default: True.",
)
@click.option(
    "--cmap",
    prompt=False,
    default="viridis",
    help="Colormap for the DEMs. Default: viridis.",
)
@click.option(
    "--downsample",
    prompt=False,
    default="auto",
    help="Downsample factor for reads, or 'auto' to size thumbnails automatically. Default: auto.",
)
@click.option(
    "--max_filesize_mb",
    prompt=False,
    default=10.0,
    type=float,
    help="Soft cap on the output PNG size in MB; auto dpi is reduced to respect it. Default: 10.",
)
@click.option(
    "--title",
    prompt=False,
    default=None,
    help="Figure suptitle. Default: none.",
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
    help="Filename for the output plot. Default: Directory name with _gallery.png suffix.",
)
def main(
    directory,
    pattern,
    files,
    hillshade,
    cmap,
    downsample,
    max_filesize_mb,
    title,
    output_directory,
    output_filename,
):
    """
    Generate a gallery figure of many DEMs sharing a single color scale.

    Provide either a --directory (searched with --pattern) or an explicit list
    of FILES. Explicit files take precedence over the directory + pattern.
    """
    directory = os.path.expanduser(directory)
    if output_directory:
        output_directory = os.path.expanduser(output_directory)

    # "auto" downsample passes through as a string; an explicit integer is cast.
    if downsample != "auto":
        downsample = int(downsample)

    if files:
        print(f"\nBuilding gallery from {len(files)} explicit file(s)\n")
        plotter = GalleryPlotter(sorted(files), downsample=downsample, title=title)
        # Default output location is the directory of the first file.
        default_dir = os.path.dirname(os.path.abspath(files[0]))
        dir_name = os.path.split(default_dir.rstrip("/\\"))[-1]
    else:
        print(f"\nBuilding gallery from '{pattern}' in {directory}\n")
        plotter = GalleryPlotter.from_directory(
            directory, pattern=pattern, downsample=downsample, title=title
        )
        default_dir = directory
        dir_name = os.path.split(directory.rstrip("/\\"))[-1]

    if output_directory is None:
        output_directory = default_dir

    if output_filename is None:
        output_filename = f"{dir_name}_gallery.png"

    os.makedirs(output_directory, exist_ok=True)

    plotter.plot_gallery(
        hillshade=hillshade,
        cmap=cmap,
        max_filesize_mb=max_filesize_mb,
        save_dir=output_directory,
        fig_fn=output_filename,
    )

    print(
        f"\nGallery plot saved to {os.path.join(output_directory, output_filename)}\n"
    )


if __name__ == "__main__":
    main()
