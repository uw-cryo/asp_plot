import logging
import os

import matplotlib.pyplot as plt

from asp_plot.utils import Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ScenePlotter(Plotter):
    """
    Plot raw or map-projected images from ASP stereo processing.

    This class extends the base Plotter class to provide visualization
    of raw or map-projected images created during ASP stereo processing.
    It locates and plots the left and right scenes from a stereo pair.

    Attributes
    ----------
    directory : str
        Root directory of ASP processing
    stereo_directory : str
        Subdirectory containing stereo outputs
    full_stereo_directory : str
        Full path to stereo directory
    left_scene_sub_fn : str or None
        Path to the left subsampled file
    right_scene_sub_fn : str or None
        Path to the right subsampled file
    title : str
        Plot title, inherited from Plotter class

    Examples
    --------
    >>> scene_plotter = ScenePlotter('/path/to/asp', 'stereo', title="Stereo Images")
    >>> scene_plotter.plot_scenes(save_dir='/path/to/output', fig_fn='stereo_images.png')
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
        This constructor attempts to locate the left and right
        subsampled image files in the stereo directory. These files are
        generated during ASP stereo processing with names ending in
        "-L_sub.tif" and "-R_sub.tif" respectively.
        """
        super().__init__(**kwargs)
        self.directory = directory
        self.stereo_directory = stereo_directory
        self.full_stereo_directory = os.path.join(directory, stereo_directory)

        self.left_scene_sub_fn = glob_file(self.full_stereo_directory, "*-L_sub.tif")
        self.right_scene_sub_fn = glob_file(self.full_stereo_directory, "*-R_sub.tif")

    def plot_scenes(self, save_dir=None, fig_fn=None):
        """
        Plot the left and right images side by side.

        Creates a figure with two subplots showing the left and right
        images from the stereo pair. Map-projection is not assumed.
        Each image is displayed with its filename above it.

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
        If either image file is missing, the corresponding subplot
        will display a message indicating that required files are missing.
        """
        if self.title is None:
            self.title = "Stereo Scenes"

        left_scene = Raster(self.left_scene_sub_fn)
        transform = left_scene.transform

        if transform is None:
            subtitle = "\nRaw Scenes, No Map-projection"
        else:
            subtitle = "\nMap-projected Scenes"

        fig, axa = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        fig.suptitle(f"{self.title}{subtitle}", size=10)
        axa = axa.ravel()

        left_scene_ma = left_scene.read_array()
        self.plot_array(ax=axa[0], array=left_scene_ma, cmap="gray", add_cbar=False)
        axa[0].set_title(f"Left\n{os.path.basename(self.left_scene_sub_fn)}", size=8)

        right_scene_ma = Raster(self.right_scene_sub_fn).read_array()
        self.plot_array(ax=axa[1], array=right_scene_ma, cmap="gray", add_cbar=False)
        axa[1].set_title(f"Right\n{os.path.basename(self.right_scene_sub_fn)}", size=8)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
