import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class StereoGeometryPlotter(StereopairMetadataParser):
    """
    Create visualizations of stereo geometry for satellite imagery.

    This class extends StereopairMetadataParser to provide plotting capabilities
    for stereo geometry visualization, including skyplots showing satellite viewing
    angles and map views showing footprints and satellite positions.

    Attributes
    ----------
    directory : str
        Path to directory containing XML files
    add_basemap : bool
        Whether to add a basemap to map plots, default is True
    image_list : list
        List of XML files found in the directory (inherited from StereopairMetadataParser)

    Examples
    --------
    >>> plotter = StereoGeometryPlotter('/path/to/stereo/directory')
    >>> plotter.dg_geom_plot(save_dir='/path/to/output', fig_fn='stereo_geom.png')
    """

    def __init__(self, directory, add_basemap=True, **kwargs):
        """
        Initialize the StereoGeometryPlotter.

        Parameters
        ----------
        directory : str
            Path to directory containing XML camera model files
        add_basemap : bool, optional
            Whether to add a contextily basemap to the plots, default is True
        **kwargs
            Additional keyword arguments passed to StereopairMetadataParser
        """
        super().__init__(directory=directory, **kwargs)
        self.add_basemap = add_basemap

    def get_scene_string(self, p, key="catid1_dict"):
        """
        Format scene metadata as a string.

        Creates a formatted string with key metadata for a scene, including
        catalog ID, GSD, viewing angles, and acquisition parameters.

        Parameters
        ----------
        p : dict
            Stereo pair dictionary containing scene metadata
        key : str, optional
            Key for the scene dictionary within the pair dictionary,
            default is "catid1_dict"

        Returns
        -------
        str
            Formatted string with scene metadata
        """
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
        """
        Generate a comprehensive title for stereo geometry plots.

        Creates a multi-line title string that includes stereo pair name,
        acquisition time information, stereo geometry parameters, and
        key metadata for both scenes.

        Parameters
        ----------
        p : dict
            Stereo pair dictionary containing metadata

        Returns
        -------
        str
            Formatted multi-line title string

        Notes
        -----
        The title includes pairname, center datetime, time offset,
        convergence angle, base-to-height ratio, bisector elevation angle,
        asymmetry angle, intersection area, and metadata for both scenes.
        """
        title = p["pairname"]
        title += "\nCenter datetime: %s" % p["cdate"]
        title += "\nTime offset: %s" % str(p["dt"])
        title += (
            "\nConv. angle: %0.2f, B:H ratio: %0.2f, BIE: %0.2f, Assym Angle: %0.2f, Int. area: %0.2f km2"
            % (
                p["conv_ang"],
                p["bh"],
                p["bie"],
                p["asymmetry_angle"],
                p["intersection_area"],
            )
        )
        title += self.get_scene_string(p, "catid1_dict")
        title += self.get_scene_string(p, "catid2_dict")
        return title

    def skyplot(self, ax, p, title=True, tight_layout=True):
        """
        Create a polar plot showing satellite viewing geometry.

        This plot shows the satellite azimuth and elevation angles for both images
        in a stereo pair on a polar plot, where azimuth is the angle and
        (90 - elevation) is the radius.

        Parameters
        ----------
        ax : matplotlib.axes.PolarAxes
            Polar axes object to plot the skyplot on
        p : dict
            Stereo pair dictionary with metadata from get_pair_dict()
        title : bool, optional
            Whether to add a title to the plot, default is True
        tight_layout : bool, optional
            Whether to apply tight layout to the figure, default is True

        Returns
        -------
        None
            Modifies the provided axes object in-place

        Notes
        -----
        In the polar plot:
        - The origin represents 90° elevation (satellite directly overhead)
        - The outer edge represents 40° elevation (50° from zenith)
        - Azimuth is measured clockwise from North (0°)
        - The symbols represent the satellite positions for each image
        - The dashed line connects the two satellite positions
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
        Create a map view of satellite paths and image footprints.

        Plots the satellite ground tracks (ephemeris) and the image footprints
        for both images in a stereo pair on a map.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes object to plot on
        p : dict
            Stereo pair dictionary with metadata from get_pair_dict()
        map_crs : str, optional
            Coordinate reference system for the map, default is "EPSG:3857" (Web Mercator)
        title : bool, optional
            Whether to add a title to the plot, default is True
        tight_layout : bool, optional
            Whether to apply tight layout to the figure, default is True

        Returns
        -------
        None
            Modifies the provided axes object in-place

        Notes
        -----
        - Satellite footprints are shown as polygons
        - Satellite paths (ephemeris) are shown as point tracks
        - The start of each satellite path is marked with a white circle
        - A basemap is added if self.add_basemap is True

        Credit
        ------
        Adapted from David Shean's notebook:
        https://github.com/dshean/dgtools/blob/master/notebooks/dg_pair_geom_eph_analysis.ipynb
        """
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

        if self.add_basemap:
            import contextily as ctx

            ctx.add_basemap(ax, crs=map_crs, attribution=False)

        ax.legend(loc="best", prop={"size": 6})
        if title:
            ax.set_title(self.get_title(p), fontsize=7.5)
        if tight_layout:
            plt.tight_layout()

    def dg_geom_plot(self, save_dir=None, fig_fn=None):
        """
        Create a comprehensive stereo geometry visualization.

        Generates a figure with two subplots:
        1. A skyplot showing satellite viewing angles (left)
        2. A map view showing satellite paths and image footprints (right)

        Parameters
        ----------
        save_dir : str, optional
            Directory to save the figure, default is None (figure not saved)
        fig_fn : str, optional
            Filename for the figure, default is None (figure not saved)

        Returns
        -------
        matplotlib.figure.Figure
            The created figure object (not shown automatically)

        Notes
        -----
        If both save_dir and fig_fn are provided, the figure is saved using
        the save_figure utility function.

        The map uses a local transverse Mercator projection centered on the
        intersection of the two image footprints to minimize distortion.
        """
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
