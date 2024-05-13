import os
import glob
import matplotlib.pyplot as plt
from dgtools.lib import dglib
from asp_plot.utils import Raster, Plotter


class ScenePlotter(Plotter):
    def __init__(self, directory, stereo_directory, **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.stereo_directory = stereo_directory

        try:
            self.left_ortho_sub_fn = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*-L_sub.tif")
            )[0]
            self.right_ortho_sub_fn = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*-R_sub.tif")
            )[0]
        except:
            raise ValueError(
                "Could not find L-sub and R-sub images in stereo directory"
            )

    def get_names_and_gsd(self):
        left_name, right_name = self.left_ortho_sub_fn.split("/")[-1].split("_")[2:4]
        right_name = right_name.split("-")[0]

        gsds = []
        for image in [left_name, right_name]:
            xml_fn = glob.glob(os.path.join(self.directory, f"{image}*.xml"))[0]
            gsd = dglib.getTag(xml_fn, "MEANPRODUCTGSD")
            if gsd is None:
                gsd = dglib.getTag(xml_fn, "MEANCOLLECTEDGSD")
            gsds.append(round(float(gsd), 2))

        scene_dict = {
            "left_name": left_name,
            "right_name": right_name,
            "left_gsd": gsds[0],
            "right_gsd": gsds[1],
        }

        return scene_dict

    def plot_orthos(self):
        scene_dict = self.get_names_and_gsd()

        f, axa = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        f.suptitle(self.title, size=10)
        axa = axa.ravel()

        ortho_ma = Raster(self.left_ortho_sub_fn).read_array()
        self.plot_array(ax=axa[0], array=ortho_ma, cmap="gray", add_cbar=False)
        axa[0].set_title(
            f"Left image\n{scene_dict['left_name']}, {scene_dict['left_gsd']:0.2f} m"
        )

        ortho_ma = Raster(self.right_ortho_sub_fn).read_array()
        self.plot_array(ax=axa[1], array=ortho_ma, cmap="gray", add_cbar=False)
        axa[1].set_title(
            f"Right image\n{scene_dict['right_name']}, {scene_dict['right_gsd']:0.2f} m"
        )

        f.tight_layout()
        plt.show()
