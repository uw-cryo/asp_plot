import logging
import os

import matplotlib.pyplot as plt

from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ScenePlotter(Plotter):
    def __init__(self, directory, stereo_directory, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.stereo_directory = stereo_directory
        self.full_stereo_directory = os.path.join(directory, stereo_directory)

        self.left_ortho_sub_fn = glob_file(self.full_stereo_directory, "*-L_sub.tif")
        self.right_ortho_sub_fn = glob_file(self.full_stereo_directory, "*-R_sub.tif")

    def plot_orthos(self, save_dir=None, fig_fn=None):
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
