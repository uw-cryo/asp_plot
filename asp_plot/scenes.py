import logging
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt

from asp_plot.utils import (
    Plotter,
    Raster,
    describe_pair,
    detect_satellite_attribution,
    find_pair_directories,
    glob_file,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class PairSceneFiles:
    """The left/right sub-sampled scenes of one multi-view pair.

    One entry per ``<prefix>-pairN/`` subdirectory of an ASP multi-view stereo
    run (see :func:`asp_plot.utils.find_pair_directories`).
    """

    number: int
    directory: str
    left_scene_sub_fn: Optional[str]
    right_scene_sub_fn: Optional[str]
    label: str


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
    pairs : list of PairSceneFiles
        Per-pair scene files of a multi-view run, resolved from the
        ``<prefix>-pairN/`` subdirectories when present; empty for a
        standard stereo run.
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
            ``*-R_sub.tif``), or in ``<prefix>-pairN/`` subdirectories for a
            multi-view run.
        """
        self.directory = os.path.expanduser(directory)
        self.stereo_directory = stereo_directory
        self.full_stereo_directory = os.path.join(self.directory, stereo_directory)

        self.attribution = detect_satellite_attribution(self.directory)

        # A multi-view run keeps its sub-sampled scenes in the per-pair
        # subdirectories instead of the top level of the stereo directory,
        # so their absence up top is expected (quiet) in that layout.
        pair_directories = find_pair_directories(self.full_stereo_directory)
        quiet = bool(pair_directories)

        self.left_scene_sub_fn = glob_file(
            self.full_stereo_directory, "*-L_sub.tif", quiet=quiet
        )
        self.right_scene_sub_fn = glob_file(
            self.full_stereo_directory, "*-R_sub.tif", quiet=quiet
        )

        # The presence of <prefix>-pairN/ subdirectories is the multi-view
        # marker (matching StereoFiles): pairs win over any stale top-level
        # sub-sampled scenes left behind by an earlier two-image run.
        self.pairs = [
            PairSceneFiles(
                number=number,
                directory=pair_directory,
                left_scene_sub_fn=glob_file(pair_directory, "*-L_sub.tif"),
                right_scene_sub_fn=glob_file(pair_directory, "*-R_sub.tif"),
                label=describe_pair(number, pair_directory),
            )
            for number, pair_directory in pair_directories
        ]


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

    @property
    def pairs(self):
        return self.files.pairs

    def _plot_scene_figure(
        self, left_fn, right_fn, suptitle, left_label="Left", save_dir=None, fig_fn=None
    ):
        """One two-panel left/right scene figure; placeholders when missing."""
        left_scene = Raster(left_fn) if left_fn else None

        if left_scene is None:
            subtitle = ""
        elif left_scene.transform is None:
            subtitle = "\nRaw Scenes, No Map-projection"
        else:
            subtitle = "\nMap-projected Scenes"

        fig, axa = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        fig.suptitle(f"{suptitle}{subtitle}", size=10)
        axa = axa.ravel()

        if left_scene is not None:
            self.plot_array(
                ax=axa[0],
                array=left_scene.read_array(),
                cmap="gray",
                add_cbar=False,
                copyright=True,
            )
            axa[0].set_title(f"{left_label}\n{os.path.basename(left_fn)}", size=8)
        else:
            self.plot_missing(axa[0])
            axa[0].set_title(left_label, size=8)

        if right_fn:
            self.plot_array(
                ax=axa[1],
                array=Raster(right_fn).read_array(),
                cmap="gray",
                add_cbar=False,
                copyright=True,
            )
            axa[1].set_title(f"Right\n{os.path.basename(right_fn)}", size=8)
        else:
            self.plot_missing(axa[1])
            axa[1].set_title("Right", size=8)

        self.save(fig, save_dir, fig_fn)

    def plot_scenes(self, save_dir=None, fig_fn=None):
        """
        Plot the left and right images side by side.

        Creates a figure with two subplots showing the left and right
        images from the stereo pair. Map-projection is not assumed.
        Each image is displayed with its filename above it.

        For a multi-view run (per-pair products in ``<prefix>-pairN/``
        subdirectories), one figure per pair is created instead, named
        ``<stem>_pairN.png``. The reference (left) image is the same for
        every pair.

        Parameters
        ----------
        save_dir : str or None, optional
            Directory to save the figure(s), default is None (don't save)
        fig_fn : str or None, optional
            Filename for the saved figure, default is None. For a multi-view
            run this is the stem the per-pair filenames are derived from.

        Returns
        -------
        list of str
            The filename(s) saved (empty if save_dir/fig_fn were not
            provided).

        Notes
        -----
        If either image file is missing, the corresponding subplot
        will display a message indicating that required files are missing.
        """
        if self.title is None:
            self.title = "Stereo Scenes"

        if self.pairs:
            stem, ext = os.path.splitext(fig_fn) if fig_fn else ("", ".png")
            saved = []
            for pair in self.pairs:
                pair_fn = f"{stem}_pair{pair.number}{ext}" if fig_fn else None
                self._plot_scene_figure(
                    pair.left_scene_sub_fn,
                    pair.right_scene_sub_fn,
                    f"{self.title} — {pair.label}",
                    left_label="Left (reference)",
                    save_dir=save_dir,
                    fig_fn=pair_fn,
                )
                if save_dir and pair_fn:
                    saved.append(pair_fn)
            return saved

        # The sub-sampled scenes may be absent entirely; plot placeholders
        # instead of crashing so the rest of a report can still be built.
        self._plot_scene_figure(
            self.left_scene_sub_fn,
            self.right_scene_sub_fn,
            self.title,
            save_dir=save_dir,
            fig_fn=fig_fn,
        )
        return [fig_fn] if save_dir and fig_fn else []
