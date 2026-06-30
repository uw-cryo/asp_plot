import logging
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

from asp_plot.csm_io import estim_satellite_orientation
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class StereoGeometryPlotter:
    """
    Create visualizations of stereo geometry for satellite imagery.

    This class *composes* a :class:`StereopairMetadataParser` to provide plotting
    capabilities for stereo geometry visualization, including skyplots showing
    satellite viewing angles and map views showing footprints and satellite
    positions. Metadata access goes through the ``parser`` attribute rather than
    inheritance, keeping the (sensor-agnostic) plotting concerns separate from
    the (sensor-specific) metadata parsing.

    Attributes
    ----------
    directory : str
        Path to directory containing XML files
    parser : StereopairMetadataParser
        The composed metadata parser used to extract stereo-pair geometry
    add_basemap : bool
        Whether to add a basemap to map plots, default is True

    Examples
    --------
    >>> plotter = StereoGeometryPlotter('/path/to/stereo/directory')
    >>> plotter.dg_geom_plot(save_dir='/path/to/output', fig_fn='stereo_geom.png')
    """

    def __init__(self, directory=None, add_basemap=True, inputs=None, **kwargs):
        """
        Initialize the StereoGeometryPlotter.

        Parameters
        ----------
        directory : str, optional
            Path to directory containing XML camera model files.
        add_basemap : bool, optional
            Whether to add a contextily basemap to the plots, default is True
        inputs : str or os.PathLike or iterable of those, optional
            Explicit files, directories, and/or glob patterns to use instead of
            a single ``directory`` (e.g. a ``geom_plot *.XML`` call). Takes
            precedence when both are given.
        **kwargs
            Additional keyword arguments passed to StereopairMetadataParser
        """
        self.parser = StereopairMetadataParser(
            directory=directory, inputs=inputs, **kwargs
        )
        # When built from an explicit input list, fall back to the parser's
        # resolved base directory (used for default output paths).
        self.directory = directory if directory is not None else self.parser.directory
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
        # Asymmetry angle and intersection area are only defined when the two
        # footprints overlap (common to miss in N-scene sets); fall back to "N/A"
        # so non-overlapping pairs still render a title. For overlapping pairs the
        # f"{x:0.2f}" formatting matches the previous "%0.2f" output exactly.
        asym = p.get("asymmetry_angle")
        asym_str = f"{asym:0.2f}" if asym is not None else "N/A"
        area = p.get("intersection_area")
        area_str = f"{area:0.2f}" if area is not None else "N/A"

        title = p["pairname"]
        title += "\nCenter datetime: %s" % p["cdate"]
        title += "\nTime offset: %s" % str(p["dt"])
        title += (
            "\nConv. angle: %0.2f, B:H ratio: %0.2f, BIE: %0.2f, Assym Angle: %s, Int. area: %s km2"
            % (
                p["conv_ang"],
                p["bh"],
                p["bie"],
                asym_str,
                area_str,
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

        # Keep the full (autoscaled) extent so both satellite tracks are visible --
        # the tracks matter more than zooming in on the footprints, and off-nadir
        # scenes have tracks tens to hundreds of km from the ground. _add_basemap_safe
        # keeps the wide-extent basemap fetch from failing on a negative auto-zoom.
        if self.add_basemap:
            self._add_basemap_safe(ax, map_crs)

        ax.legend(loc="best", prop={"size": 6})
        if title:
            ax.set_title(self.get_title(p), fontsize=7.5)
        if tight_layout:
            plt.tight_layout()

    @staticmethod
    def _scene_colors(n):
        """Return n distinct colors for scene/footprint coloring (tab10 cycle)."""
        cmap = plt.get_cmap("tab10")
        return [cmap(i % 10) for i in range(n)]

    def _render_pair(self, p, save_dir=None, fig_fn=None):
        """
        Render one stereo-pair figure (skyplot + map + full-stats title).

        This is the per-pair figure shared by the two-scene case and every pair of
        an N-scene set.

        Parameters
        ----------
        p : dict
            Stereo-pair dictionary from the parser.
        save_dir, fig_fn : str, optional
            If both are given, the figure is saved there.
        """
        fig = plt.figure(figsize=(10, 7.5))
        G = gridspec.GridSpec(nrows=1, ncols=2)
        ax0 = fig.add_subplot(G[0, 0:1], polar=True)
        ax1 = fig.add_subplot(G[0, 1:2])

        self.skyplot(ax0, p, title=False, tight_layout=False)

        # Use local projection to minimize distortion; transverse mercator is fine
        # here (usually within ~2-3 deg). Robust to non-overlapping pairs.
        map_crs = self.parser.get_pair_map_projection(p, proj_type="tmerc")

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
            # Saved to disk; release it so N-scene runs don't accumulate dozens of
            # open figures. Interactive callers (no save) keep the figure to display.
            plt.close(fig)

    def _overview_skyplot(self, ax, scenes):
        """Skyplot of all N satellite positions, color-coded by scene."""
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location("N")
        ax.grid(True)

        plot_kw = {"marker": "o", "ls": "", "ms": 5}
        ax.plot(0, 0, marker="o", color="k")
        for d, color in zip(scenes, self._scene_colors(len(scenes))):
            ax.plot(
                np.radians(d["meansataz"]),
                (90 - d["meansatel"]),
                label=d.get("catid", "?"),
                color=color,
                **plot_kw,
            )
        ax.legend(loc="lower left", fontsize="small")
        ax.set_rmin(0)
        ax.set_rmax(50)

    def _overview_map(self, ax, scenes, map_crs):
        """Map of all N footprints and ephemeris tracks, color-coded by scene.

        Unlike the per-pair map, the overview keeps the full (autoscaled) extent so
        every satellite track is visible -- off-nadir scenes have tracks tens to
        hundreds of km from the ground footprints, and the overview's purpose is to
        show that multi-view geometry. Footprints are drawn first and the tracks on
        top, so the tracks are not buried under the (semi-transparent) overlapping
        footprints.
        """
        poly_kw = {"alpha": 0.5, "edgecolor": "k", "linewidth": 0.5}
        eph_kw = {"markersize": 2}
        start_kw = {"markersize": 5, "facecolor": "w", "edgecolor": "k"}
        colors = self._scene_colors(len(scenes))

        # Footprints on the bottom...
        for d, color in zip(scenes, colors):
            d["fp_gdf"].to_crs(map_crs).plot(ax=ax, color=color, **poly_kw)
        # ...then every satellite track and its start marker on top.
        for d, color in zip(scenes, colors):
            eph_gdf = d["eph_gdf"]
            eph_gdf.to_crs(map_crs).plot(
                ax=ax, label=d.get("catid", "?"), color=color, **eph_kw
            )
            eph_gdf.iloc[0:2].to_crs(map_crs).plot(ax=ax, **start_kw)

        if self.add_basemap:
            self._add_basemap_safe(ax, map_crs)

        ax.legend(loc="best", prop={"size": 6})

    @staticmethod
    def _add_basemap_safe(ax, map_crs):
        """Add a contextily basemap, clamped for wide (overview) extents.

        Very large extents make contextily's auto-zoom resolve to an invalid
        (negative) zoom that errors; fall back to a coarse continental zoom so the
        overview still gets a basemap instead of silently dropping it.
        """
        import contextily as ctx

        try:
            ctx.add_basemap(ax, crs=map_crs, attribution=False)
        except Exception:
            try:
                ctx.add_basemap(ax, crs=map_crs, attribution=False, zoom=6)
            except Exception:
                pass

    def _overview_title(self, scenes):
        """Acquisition summary title for the overview figure (no pair metrics)."""
        base = os.path.split(self.parser.directory.rstrip("/\\"))[-1]
        catids = [d.get("catid", "?") for d in scenes]
        title = f"{base}: {len(scenes)}-scene overview"
        dates = sorted(d["date"] for d in scenes if d.get("date") is not None)
        if dates:
            title += f"\n{dates[0]} to {dates[-1]}"
        title += "\n" + ", ".join(catids)
        return title

    def _render_overview(self, scenes, save_dir=None, fig_fn=None):
        """Render the N-scene overview figure (all scenes, color-coded)."""
        fig = plt.figure(figsize=(10, 7.5))
        G = gridspec.GridSpec(nrows=1, ncols=2)
        ax0 = fig.add_subplot(G[0, 0:1], polar=True)
        ax1 = fig.add_subplot(G[0, 1:2])

        self._overview_skyplot(ax0, scenes)
        map_crs = self.parser.get_scenes_centroid_projection(proj_type="tmerc")
        self._overview_map(ax1, scenes, map_crs)

        plt.suptitle(self._overview_title(scenes), fontsize=10)

        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
            plt.close(fig)

    def dg_geom_plot(self, save_dir=None, fig_fn=None):
        """
        Create stereo geometry visualization(s).

        For exactly two scenes this produces a single figure (skyplot + map +
        full pairwise-stats title) — unchanged from prior behavior. For more than
        two scenes it produces, as separate files:

        - one **overview** figure with all scenes color-coded (skyplot + map), and
        - one figure **per pair** (every N-choose-2 combination), each with the
          full pairwise stats in its title.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save the figure(s), default is None (not saved).
        fig_fn : str, optional
            Filename for the two-scene figure; for N>2 it is the stem from which
            ``<stem>_overview.png`` and ``<stem>_<labelA>_<labelB>.png`` are
            derived (``label`` is the CATID, or ``pairN`` if a CATID is missing).

        Returns
        -------
        list of str
            The filename(s) saved (empty if save_dir/fig_fn were not provided).

        Notes
        -----
        Each figure's map uses a local transverse Mercator projection centered on
        the relevant footprints to minimize distortion.
        """
        scenes = self.parser.get_catid_dicts()
        n = len(scenes)
        if n < 2:
            raise ValueError(
                f"Need at least two scenes for stereo geometry, but found {n}."
            )

        if n == 2:
            p = self.parser.get_pair_dict()
            self._render_pair(p, save_dir, fig_fn)
            return [fig_fn] if (save_dir and fig_fn) else []

        # N > 2: overview figure + one figure per pair.
        stem, ext = os.path.splitext(fig_fn) if fig_fn else ("stereo_geom", ".png")
        saved = []

        overview_fn = f"{stem}_overview{ext}" if fig_fn else None
        self._render_overview(scenes, save_dir, overview_fn)
        if save_dir and overview_fn:
            saved.append(overview_fn)

        for i, p in enumerate(self.parser.get_pair_dicts()):
            c1 = p["catid1_dict"].get("catid")
            c2 = p["catid2_dict"].get("catid")
            suffix = f"{c1}_{c2}" if (c1 and c2) else f"pair{i + 1}"
            pair_fn = f"{stem}_{suffix}{ext}" if fig_fn else None
            self._render_pair(p, save_dir, pair_fn)
            if save_dir and pair_fn:
                saved.append(pair_fn)

        return saved

    @staticmethod
    def _compute_roll_pitch_yaw(eph_gdf, att_df):
        """
        Compute roll, pitch, yaw from attitude quaternions relative to orbital frame.

        Uses ECEF positions from ephemeris to estimate the nominal nadir-pointing
        orientation (LVLH frame), then computes the relative rotation from that
        reference to the actual body attitude reported by the quaternions.

        Parameters
        ----------
        eph_gdf : geopandas.GeoDataFrame
            Ephemeris GeoDataFrame with x, y, z columns in ECEF
        att_df : pandas.DataFrame
            Attitude DataFrame with q1, q2, q3, q4 columns

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 3) with roll, pitch, yaw in degrees
        """
        # Use the minimum length in case ephemeris and attitude have different counts
        n = min(len(eph_gdf), len(att_df))
        positions = eph_gdf[["x", "y", "z"]].values[:n].tolist()
        ref_rotations = estim_satellite_orientation(positions)

        rpy = np.zeros((n, 3))
        for i in range(n):
            body_rot = R.from_quat(
                att_df[["q1", "q2", "q3", "q4"]].iloc[i].values.copy()
            ).as_matrix()
            ref_inv = np.linalg.inv(ref_rotations[i])
            relative = np.matmul(ref_inv, body_rot)
            rpy[i] = R.from_matrix(relative).as_euler("XYZ", degrees=True)

        return rpy

    def satellite_position_orientation_plot(self, save_dir=None, fig_fn=None):
        """
        Create a visualization of satellite position and orientation data.

        Generates a 3-row x 2-column figure (one column per scene):
        - Row 0: Map of satellite positions colored by position covariance std
        - Row 1: Roll, pitch, yaw relative to orbital reference frame over time
        - Row 2: Attitude covariance trace std over time

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
        """
        p = self.parser.get_pair_dict()

        map_crs = self.parser.get_centroid_projection(
            p["intersection"], proj_type="tmerc"
        )

        fig = plt.figure(figsize=(14, 12))
        G = gridspec.GridSpec(nrows=3, ncols=2, hspace=0.35, wspace=0.3)

        catid_keys = ["catid1_dict", "catid2_dict"]
        c_list = ["blue", "orange"]

        for col, key in enumerate(catid_keys):
            d = p[key]
            eph_gdf = d["eph_gdf"]
            att_df = d["att_df"]
            catid = d["catid"]

            # Row 0: Position map colored by position covariance std
            ax0 = fig.add_subplot(G[0, col])
            pos_cov_std = np.sqrt(
                eph_gdf["cov_11"] + eph_gdf["cov_22"] + eph_gdf["cov_33"]
            )
            eph_gdf_proj = eph_gdf.to_crs(map_crs)
            sc = ax0.scatter(
                eph_gdf_proj.geometry.x,
                eph_gdf_proj.geometry.y,
                c=pos_cov_std,
                s=5,
                cmap="viridis",
            )
            fp_gdf = d["fp_gdf"]
            fp_gdf.to_crs(map_crs).plot(
                ax=ax0, facecolor="none", edgecolor=c_list[col], linewidth=1
            )
            if self.add_basemap:
                try:
                    import contextily as ctx

                    ctx.add_basemap(ax0, crs=map_crs, attribution=False)
                except Exception:
                    pass
            fig.colorbar(sc, ax=ax0, label="Position std (m)", shrink=0.8)
            ax0.set_title(f"{catid}\nPosition Covariance", fontsize=9)
            ax0.tick_params(labelsize=7)

            # Row 1: Roll, pitch, yaw relative to orbital frame
            ax1 = fig.add_subplot(G[1, col])
            n = min(len(eph_gdf), len(att_df))
            time_seconds = (att_df.index[:n] - att_df.index[0]).total_seconds()
            rpy = self._compute_roll_pitch_yaw(eph_gdf, att_df)
            for i, label in enumerate(["Roll", "Pitch", "Yaw"]):
                ax1.plot(time_seconds, rpy[:, i], label=label, linewidth=0.8)
            ax1.set_xlabel("Time (s)", fontsize=8)
            ax1.set_ylabel("Angle (deg)", fontsize=8)
            ax1.legend(fontsize=7, loc="best")
            ax1.set_title(f"{catid}\nRoll / Pitch / Yaw", fontsize=9)
            ax1.tick_params(labelsize=7)

            # Row 2: Attitude covariance trace std over time
            ax2 = fig.add_subplot(G[2, col])
            time_seconds_full = (att_df.index - att_df.index[0]).total_seconds()
            att_cov_std = np.sqrt(
                att_df["cov_11"]
                + att_df["cov_22"]
                + att_df["cov_33"]
                + att_df["cov_44"]
            )
            ax2.plot(time_seconds_full, att_cov_std, color=c_list[col], linewidth=0.8)
            ax2.set_xlabel("Time (s)", fontsize=8)
            ax2.set_ylabel("Attitude std (trace)", fontsize=8)
            ax2.set_title(f"{catid}\nAttitude Covariance Trace", fontsize=9)
            ax2.tick_params(labelsize=7)

        plt.suptitle(f"{p['pairname']}\nSatellite Position & Orientation", fontsize=11)

        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
