import logging
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SceneGeometryPlotter(StereopairMetadataParser):
    def __init__(self, directory, **kwargs):
        super().__init__(directory=directory, **kwargs)

    def get_scene_string(self, p, key="catid1_dict"):
        scene_string = (
            "\nID:%s, GSD:%0.2f, off:%0.1f, az:%0.1f, el:%0.1f, it:%0.1f, ct:%0.1f, scan:%s, tdi:%i"
            % (
                p[key]["catid"],
                p[key]["meanproductgsd"],
                p[key]["meanoffnadirviewangle"],
                p[key]["meansataz"],
                (90 - p[key]["meansatel"]),
                p[key]["meanintrackviewangle"],
                (p[key]["meancrosstrackviewangle"]),
                p[key]["scandir"],
                p[key]["tdi"],
            )
        )
        return scene_string

    def get_title(self, p):
        title = p["pairname"]
        title += "\nCenter datetime: %s" % p["cdate"]
        title += "\nTime offset: %s" % str(p["dt"])
        title += "\nConv. angle: %0.2f, B:H ratio: %0.2f, Int. area: %0.2f km2" % (
            p["conv_ang"],
            p["bh"],
            p["intersection_area"],
        )
        title += self.get_scene_string(p, "catid1_dict")
        title += self.get_scene_string(p, "catid2_dict")
        return title

    def skyplot(self, ax, p, title=True, tight_layout=True):
        """
        Function to plot stereo geometry from dg xml
        Parameters
        -----------
        p: pair dictionary
            dictionary with xml info read from get_pair_dict function
        ax: matplotlib.axes
            polar axes object to plot the skyplot on
        """
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.grid(True)

        plot_kw = {"marker": "o", "ls": "", "ms": 5}

        ax.plot(0, 0, marker="o", color="k")
        ax.plot(
            np.radians(p["catid1_dict"]["meansataz"]),
            (90 - p["catid1_dict"]["meansatel"]),
            label=p["catid1_dict"]["catid"],
            **plot_kw,
        )
        ax.plot(
            np.radians(p["catid2_dict"]["meansataz"]),
            (90 - p["catid2_dict"]["meansatel"]),
            label=p["catid2_dict"]["catid"],
            **plot_kw,
        )
        ax.plot(
            [
                np.radians(p["catid1_dict"]["meansataz"]),
                np.radians(p["catid2_dict"]["meansataz"]),
            ],
            [90 - p["catid1_dict"]["meansatel"], 90 - p["catid2_dict"]["meansatel"]],
            color="k",
            ls="--",
            lw=0.5,
            alpha=0.5,
        )

        ax.legend(loc="lower left", fontsize="small")

        ax.set_rmin(0)
        ax.set_rmax(50)
        if title:
            ax.set_title(self.get_title(p), fontsize=8)
        if tight_layout:
            plt.tight_layout()

    def map_plot(self, ax, p, map_crs="EPSG:3857", title=True, tight_layout=True):
        """
        Plot satellite ephemeris and ground footprint for a DigitalGlobe stereo pair
        # stitched together from David's notebook: https://github.com/dshean/dgtools/blob/master/notebooks/dg_pair_geom_eph_analysis.ipynb
        Parameters
        ------------
        ax: matplotlib sublot axes object
        gdf_list: list of necessary GeoDataFrame objects
        """
        import contextily as ctx

        poly_kw = {"alpha": 0.5, "edgecolor": "k", "linewidth": 0.5}
        eph_kw = {"markersize": 2}

        eph1_gdf = p["catid1_dict"]["eph_gdf"]
        eph2_gdf = p["catid2_dict"]["eph_gdf"]
        fp1_gdf = p["catid1_dict"]["fp_gdf"]
        fp2_gdf = p["catid2_dict"]["fp_gdf"]

        c_list = ["blue", "orange"]
        fp1_gdf.to_crs(map_crs).plot(ax=ax, color=c_list[0], **poly_kw)
        fp2_gdf.to_crs(map_crs).plot(ax=ax, color=c_list[1], **poly_kw)
        eph1_gdf.to_crs(map_crs).plot(
            ax=ax, label=p["catid1_dict"]["catid"], color=c_list[0], **eph_kw
        )
        eph2_gdf.to_crs(map_crs).plot(
            ax=ax, label=p["catid2_dict"]["catid"], color=c_list[1], **eph_kw
        )

        start_kw = {"markersize": 5, "facecolor": "w", "edgecolor": "k"}
        eph1_gdf.iloc[0:2].to_crs(map_crs).plot(ax=ax, **start_kw)
        eph2_gdf.iloc[0:2].to_crs(map_crs).plot(ax=ax, **start_kw)

        ctx.add_basemap(ax, crs=map_crs, attribution=False)

        ax.legend(loc="best", prop={"size": 6})
        if title:
            ax.set_title(self.get_title(p), fontsize=7.5)
        if tight_layout:
            plt.tight_layout()

    def dg_geom_plot(self, save_dir=None, fig_fn=None):
        # load pair information as dict
        p = self.get_pair_dict()

        fig = plt.figure(figsize=(10, 7.5))
        G = gridspec.GridSpec(nrows=1, ncols=2)
        ax0 = fig.add_subplot(G[0, 0:1], polar=True)
        ax1 = fig.add_subplot(G[0, 1:2])

        self.skyplot(ax0, p, title=False, tight_layout=False)

        # Use local projection to minimize distortion
        # Should be OK to use transverse mercator here, usually within ~2-3 deg
        map_crs = self.get_centroid_projection(p["intersection"], proj_type="tmerc")

        self.map_plot(
            ax1,
            p,
            map_crs=map_crs,
            title=False,
            tight_layout=False,
        )

        plt.suptitle(self.get_title(p), fontsize=10)

        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)


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
