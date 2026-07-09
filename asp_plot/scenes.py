import logging
import os

import matplotlib.pyplot as plt

from asp_plot.utils import Plotter, Raster, detect_satellite_attribution, glob_file

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SceneFiles:
    """
    Discover the left/right sub-sampled scene files for a stereo directory.

    Isolates file discovery from the plotting in ``ScenePlotter``, mirroring the
    ``StereoFiles`` / ``StereoPlotter`` split. Resolved paths and flags are
    exposed as plain attributes for ``ScenePlotter`` to consume.

    Attributes
    ----------
    directory : str
        Root directory of ASP processing.
    stereo_directory : str
        Subdirectory containing stereo outputs.
    full_stereo_directory : str
        Full path to the stereo directory.
    attribution : str or None
        Rights-holder of the source imagery ("Vantor", "Airbus DS", ...);
        gates the copyright overlay on scene panels.
    left_scene_sub_fn, right_scene_sub_fn : str or None
        Paths to the left/right sub-sampled scene files.
    """

    def __init__(self, directory, stereo_directory):
        """
        Discover the sub-sampled scene files.

        Parameters
        ----------
        directory : str
            Root directory of ASP processing.
        stereo_directory : str
            Subdirectory containing stereo outputs. The left and right
            sub-sampled images are located here (``*-L_sub.tif`` /
            ``*-R_sub.tif``).
        """
        self.directory = os.path.expanduser(directory)
        self.stereo_directory = stereo_directory
        self.full_stereo_directory = os.path.join(self.directory, stereo_directory)

        self.attribution = detect_satellite_attribution(self.directory)

        self.left_scene_sub_fn = glob_file(self.full_stereo_directory, "*-L_sub.tif")
        self.right_scene_sub_fn = glob_file(self.full_stereo_directory, "*-R_sub.tif")


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
        "-L_sub.tif" and "-R_sub.tif" respectively. File discovery is delegated
        to :class:`SceneFiles`; the resolved paths are exposed as read-only
        properties on this plotter.
        """
        self.files = SceneFiles(directory, stereo_directory)
        super().__init__(attribution=self.files.attribution, **kwargs)

    @property
    def directory(self):
        return self.files.directory

    @property
    def stereo_directory(self):
        return self.files.stereo_directory

    @property
    def full_stereo_directory(self):
        return self.files.full_stereo_directory

    @property
    def left_scene_sub_fn(self):
        return self.files.left_scene_sub_fn

    @property
    def right_scene_sub_fn(self):
        return self.files.right_scene_sub_fn

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

        # The sub-sampled scenes may be absent (e.g. a multi-view run keeps
        # them in its run-pair*/ subdirectories); plot placeholders instead of
        # crashing so the rest of a report can still be built.
        left_scene = Raster(self.left_scene_sub_fn) if self.left_scene_sub_fn else None

        if left_scene is None:
            subtitle = ""
        elif left_scene.transform is None:
            subtitle = "\nRaw Scenes, No Map-projection"
        else:
            subtitle = "\nMap-projected Scenes"

        fig, axa = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        fig.suptitle(f"{self.title}{subtitle}", size=10)
        axa = axa.ravel()

        if left_scene is not None:
            self.plot_array(
                ax=axa[0],
                array=left_scene.read_array(),
                cmap="gray",
                add_cbar=False,
                copyright=True,
            )
            axa[0].set_title(
                f"Left\n{os.path.basename(self.left_scene_sub_fn)}", size=8
            )
        else:
            self.plot_missing(axa[0])
            axa[0].set_title("Left", size=8)

        if self.right_scene_sub_fn:
            self.plot_array(
                ax=axa[1],
                array=Raster(self.right_scene_sub_fn).read_array(),
                cmap="gray",
                add_cbar=False,
                copyright=True,
            )
            axa[1].set_title(
                f"Right\n{os.path.basename(self.right_scene_sub_fn)}", size=8
            )
        else:
            self.plot_missing(axa[1])
            axa[1].set_title("Right", size=8)

        self.save(fig, save_dir, fig_fn)
