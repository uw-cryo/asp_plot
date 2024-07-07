import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from shapely import wkt
from asp_plot.utils import Raster, Plotter, save_figure
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser


class SceneGeometryPlotter(StereopairMetadataParser):
    def __init__(self, directory, **kwargs):
        super().__init__(directory=directory, **kwargs)

    def get_scene_string(self, p, key="id1_dict"):
        scene_string = (
            "\nID:%s, GSD:%0.2f, off:%0.1f, az:%0.1f, el:%0.1f, it:%0.1f, ct:%0.1f, scan:%s, tdi:%i"
            % (
                p[key]["id"],
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
        title += self.get_scene_string(p, "id1_dict")
        title += self.get_scene_string(p, "id2_dict")
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
            np.radians(p["id1_dict"]["meansataz"]),
            (90 - p["id1_dict"]["meansatel"]),
            label=p["id1_dict"]["id"],
            **plot_kw,
        )
        ax.plot(
            np.radians(p["id2_dict"]["meansataz"]),
            (90 - p["id2_dict"]["meansatel"]),
            label=p["id2_dict"]["id"],
            **plot_kw,
        )
        ax.plot(
            [
                np.radians(p["id1_dict"]["meansataz"]),
                np.radians(p["id2_dict"]["meansataz"]),
            ],
            [90 - p["id1_dict"]["meansatel"], 90 - p["id2_dict"]["meansatel"]],
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

    def map_plot(
        self, ax, p, gdf_list, map_crs="EPSG:3857", title=True, tight_layout=True
    ):
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

        fp1_gdf, fp2_gdf, eph1_gdf, eph2_gdf = gdf_list

        c_list = ["blue", "orange"]
        fp1_gdf.to_crs(map_crs).plot(ax=ax, color=c_list[0], **poly_kw)
        fp2_gdf.to_crs(map_crs).plot(ax=ax, color=c_list[1], **poly_kw)
        eph1_gdf.to_crs(map_crs).plot(
            ax=ax, label=p["id1_dict"]["id"], color=c_list[0], **eph_kw
        )
        eph2_gdf.to_crs(map_crs).plot(
            ax=ax, label=p["id2_dict"]["id"], color=c_list[1], **eph_kw
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

        # TODO: Should store xml names in the pairdict
        # Use r100 outputs from dg_mosaic
        xml_list = sorted(glob.glob(os.path.join(self.directory, "*r100.[Xx][Mm][Ll]")))

        eph1_gdf, eph2_gdf = [self.getEphem_gdf(xml) for xml in xml_list]
        fp1_gdf, fp2_gdf = [self.xml2gdf(xml) for xml in xml_list]

        fig = plt.figure(figsize=(10, 7.5))
        G = gridspec.GridSpec(nrows=1, ncols=2)
        ax0 = fig.add_subplot(G[0, 0:1], polar=True)
        ax1 = fig.add_subplot(G[0, 1:2])

        self.skyplot(ax0, p, title=False, tight_layout=False)

        # map_crs = 'EPSG:3857'
        # Use local projection to minimize distortion
        # Get Shapely polygon and compute centroid (for local projection def)
        p_poly = wkt.loads(p["intersection"].ExportToWkt())
        p_int_c = np.array(p_poly.centroid.coords.xy).ravel()
        # map_crs = '+proj=ortho +lon_0={} +lat_0={}'.format(*p_int_c)
        # Should be OK to use transverse mercator here, usually within ~2-3 deg
        map_crs = "+proj=tmerc +lon_0={} +lat_0={}".format(*p_int_c)

        self.map_plot(
            ax1,
            p,
            [fp1_gdf, fp2_gdf, eph1_gdf, eph2_gdf],
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

    def plot_orthos(self, save_dir=None, fig_fn=None):
        p = StereopairMetadataParser(self.directory).get_pair_dict()

        fig, axa = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
        fig.suptitle(self.title, size=10)
        axa = axa.ravel()

        ortho_ma = Raster(self.left_ortho_sub_fn).read_array()
        self.plot_array(ax=axa[0], array=ortho_ma, cmap="gray", add_cbar=False)
        axa[0].set_title(
            f"Left image\n{p['id1_dict']['id']}, {p['id1_dict']['meanproductgsd']:0.2f} m"
        )

        ortho_ma = Raster(self.right_ortho_sub_fn).read_array()
        self.plot_array(ax=axa[1], array=ortho_ma, cmap="gray", add_cbar=False)
        axa[1].set_title(
            f"Right image\n{p['id2_dict']['id']}, {p['id2_dict']['meanproductgsd']:0.2f} m"
        )

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
