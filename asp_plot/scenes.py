import logging
import os

import matplotlib.pyplot as plt

from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ScenePlotter(Plotter):
    """
    Plot orthorectified images from ASP stereo processing.

    This class extends the base Plotter class to provide visualization
    of orthorectified images created during ASP stereo processing.
    It locates and plots the left and right orthoimages from a stereo pair.

    Attributes
    ----------
    directory : str
        Root directory of ASP processing
    stereo_directory : str
        Subdirectory containing stereo outputs
    full_stereo_directory : str
        Full path to stereo directory
    left_ortho_sub_fn : str or None
        Path to the left orthoimage subsampled file
    right_ortho_sub_fn : str or None
        Path to the right orthoimage subsampled file
    title : str
        Plot title, inherited from Plotter class

    Examples
    --------
    >>> scene_plotter = ScenePlotter('/path/to/asp', 'stereo', title="Stereo Images")
    >>> scene_plotter.plot_orthos(save_dir='/path/to/output', fig_fn='ortho_images.png')
    """

    def __init__(self, directory, stereo_directory, **kwargs):
        """
        Initialize the ScenePlotter object.

        Parameters
        ----------
        directory : str
            Root directory of ASP processing
        stereo_directory : str
            Subdirectory containing stereo outputs
        **kwargs : dict, optional
            Additional keyword arguments to pass to the Plotter base class,
            particularly 'title' for the plot title

        Notes
        -----
        This constructor attempts to locate the left and right orthoimage
        subsampled files in the stereo directory. These files are typically
        generated during ASP stereo processing with names ending in
        "-L_sub.tif" and "-R_sub.tif" respectively.
        """
        super().__init__(**kwargs)
        self.directory = directory
        self.stereo_directory = stereo_directory
        self.full_stereo_directory = os.path.join(directory, stereo_directory)

        self.left_ortho_sub_fn = glob_file(self.full_stereo_directory, "*-L_sub.tif")
        self.right_ortho_sub_fn = glob_file(self.full_stereo_directory, "*-R_sub.tif")

    def plot_orthos(self, save_dir=None, fig_fn=None):
        """
        Plot the left and right orthorectified images side by side.

        Creates a figure with two subplots showing the left and right
        orthorectified images from the stereo pair. Each image is displayed
        with its catalog ID and ground sample distance (GSD).

        Parameters
        ----------
        save_dir : str or None, optional
            Directory to save the figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for the saved figure, default is None

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        If either orthoimage file is missing, the corresponding subplot
        will display a message indicating that required files are missing.
        The plot includes metadata about each image, including the catalog ID
        and ground sample distance, obtained from the stereopair metadata.
        """
        p = StereopairMetadataParser(self.directory).get_pair_dict()

        fig, axa = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        fig.suptitle(self.title, size=10)
        axa = axa.ravel()

        if self.left_ortho_sub_fn:
            ortho_ma = Raster(self.left_ortho_sub_fn).read_array()
            self.plot_array(ax=axa[0], array=ortho_ma, cmap="gray", add_cbar=False)
            axa[0].set_title(
                f"Left image\n{p['catid1_dict']['catid']}, {p['catid1_dict']['meanproductgsd']:0.2f} m"
            )
        else:
            axa[0].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[0].transAxes,
            )

        if self.right_ortho_sub_fn:
            ortho_ma = Raster(self.right_ortho_sub_fn).read_array()
            self.plot_array(ax=axa[1], array=ortho_ma, cmap="gray", add_cbar=False)
            axa[1].set_title(
                f"Right image\n{p['catid2_dict']['catid']}, {p['catid2_dict']['meanproductgsd']:0.2f} m"
            )
        else:
            axa[1].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[1].transAxes,
            )

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
