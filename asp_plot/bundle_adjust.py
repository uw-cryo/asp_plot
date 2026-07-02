import glob
import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from pyproj import Transformer
from scipy.spatial.transform import Rotation
from shapely.geometry import Point

from asp_plot.csm_io import (
    getTimeAtLine,
    isLinescan,
    read_csm_cam,
    read_positions_rotations_from_file,
)
from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import (
    ColorBar,
    Plotter,
    get_xml_tag,
    glob_file,
    run_subprocess_command,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ReadBundleAdjustFiles:
    """
    Read and process bundle adjustment output files from ASP.

    This class provides functionality to read and process the outputs
    from ASP's bundle_adjust tool, including residual point maps,
    map-projected residuals, and triangulation uncertainty. It can also
    generate geodiff files to compare bundle adjustment results with
    reference DEMs.

    Attributes
    ----------
    directory : str
        Root directory of ASP processing
    bundle_adjust_directory : str
        Subdirectory containing bundle adjustment outputs
    full_directory : str
        Full path to bundle adjustment directory

    Examples
    --------
    >>> ba_reader = ReadBundleAdjustFiles('/path/to/asp', 'ba')
    >>> initial_gdf, final_gdf = ba_reader.get_initial_final_residuals_gdfs()
    >>> geodiff_initial, geodiff_final = ba_reader.get_initial_final_geodiff_gdfs()
    >>> mapproj_gdf = ba_reader.get_mapproj_residuals_gdf()
    """

    def __init__(self, directory, bundle_adjust_directory):
        """
        Initialize the ReadBundleAdjustFiles object.

        Parameters
        ----------
        directory : str
            Root directory of ASP processing
        bundle_adjust_directory : str
            Subdirectory containing bundle adjustment outputs
        """
        self.directory = os.path.expanduser(directory)
        self.bundle_adjust_directory = bundle_adjust_directory
        self.full_directory = os.path.join(self.directory, bundle_adjust_directory)

    def get_csv_paths(self, geodiff_files=False):
        """
        Get paths to bundle adjustment residual CSV files.

        Finds the paths to the initial and final residual point map CSV files
        from bundle adjustment. Can optionally find or generate geodiff files
        comparing these residuals to a reference DEM.

        Parameters
        ----------
        geodiff_files : bool, optional
            Whether to return paths to geodiff files instead of
            regular residual CSV files, default is False

        Returns
        -------
        tuple of str
            Paths to initial and final residual CSV files or geodiff files

        Raises
        ------
        ValueError
            If required bundle adjustment CSV files cannot be found

        Notes
        -----
        If geodiff_files is True and the files don't exist, this method
        will attempt to generate them using the geodiff ASP tool.
        """
        filenames = [
            "*-initial_residuals_pointmap.csv",
            "*-final_residuals_pointmap.csv",
        ]

        if geodiff_files:
            base_paths = [glob_file(self.full_directory, f) for f in filenames]
            diff_paths = []
            for path in base_paths:
                diff_path = path.replace(".csv", "-diff.csv")
                if not os.path.exists(diff_path):
                    self.generate_geodiff(path)
                if not os.path.exists(diff_path):
                    raise ValueError(
                        f"\n\nGeodiff file {diff_path} could not be generated. "
                        "Check that geodiff is installed and a reference DEM was used in bundle_adjust.\n\n"
                    )
                diff_paths.append(diff_path)
            initial_diff, final_diff = diff_paths
            return initial_diff, final_diff

        else:
            paths = [glob_file(self.full_directory, f) for f in filenames]
            for path in paths:
                if path is None:
                    raise ValueError(
                        "\n\nInitial and final bundle adjust CSV file not found. Did you run bundle_adjust?\n\n"
                    )
            initial, final = paths
            return initial, final

    def generate_geodiff(self, path):
        """
        Generate geodiff file comparing residuals to a reference DEM.

        Uses ASP's geodiff tool to calculate height differences between
        bundle adjustment residual points and a reference DEM.

        Parameters
        ----------
        path : str
            Path to the bundle adjustment residual CSV file

        Notes
        -----
        The geodiff output file will be saved with the same name as the
        input file, but with "-diff.csv" appended. This method requires
        the geodiff tool from ASP and a reference DEM that was used in
        the bundle adjustment process.
        """
        processing_parameters = ProcessingParameters(
            processing_directory=self.directory,
            bundle_adjust_directory=self.bundle_adjust_directory,
        )
        refdem = processing_parameters.get_reference_dem(
            processing_parameters.bundle_adjust_log
        )

        if not refdem:
            logger.warning(
                f"\n\nNo reference DEM found in bundle_adjust log. This would only exist if you ran bundle_adjust with the advanced `--mapproj-dem ref_dem.tif` flag. Cannot generate geodiff for {path}.\n\n"
            )
            return

        # If the reference DEM path from the log is relative, make it absolute
        if not os.path.isabs(refdem):
            refdem = os.path.join(self.directory, refdem)

        try:
            command = [
                "geodiff",
                "--csv-format",
                "1:lon 2:lat 3:height_above_datum",
                f"{refdem}",
                f"{path}",
                "-o",
                f"{path.replace('.csv', '')}",
            ]

            run_subprocess_command(command)
        except Exception:
            logger.warning(
                f"\n\nCould not generate geodiff file for {path}. Check that the geodiff ASP tool is installed and the reference DEM {refdem} exists.\n\n"
            )

    def get_residuals_gdf(self, csv_path, residuals_in_meters=True):
        """
        Read bundle adjustment residuals into a GeoDataFrame.

        Reads a bundle adjustment residuals CSV file and converts it
        to a GeoDataFrame with appropriate geometry. Can optionally
        convert residuals from pixels to meters.

        Parameters
        ----------
        csv_path : str
            Path to the residuals CSV file
        residuals_in_meters : bool, optional
            Whether to convert residuals from pixels to meters using
            the mean GSD from the stereopair metadata, default is True

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing residual data with point geometry

        Notes
        -----
        The returned GeoDataFrame has a 'filename' attribute set to the
        base name of the input file. If residuals_in_meters is True,
        a new column 'mean_residual_meters' will be added with pixel
        residuals converted to meters.
        """
        cols = [
            "lon",
            "lat",
            "height_above_datum",
            "mean_residual",
            "num_observations",
        ]

        resid_df = pd.read_csv(csv_path, skiprows=2, names=cols)

        # Need the astype('str') to handle cases where column has dtype of int (without the # from DEM appended to some rows)
        resid_df["from_DEM"] = (
            resid_df["num_observations"].astype("str").str.contains("# from DEM")
        )

        resid_df["num_observations"] = (
            resid_df["num_observations"]
            .astype("str")
            .str.split("#", expand=True)[0]
            .astype(int)
        )

        resid_gdf = gpd.GeoDataFrame(
            resid_df,
            geometry=gpd.points_from_xy(
                resid_df["lon"], resid_df["lat"], crs="EPSG:4326"
            ),
        )

        if residuals_in_meters:
            try:
                p = StereopairMetadataParser(self.directory).get_pair_dict()
                mean_gsd = np.average(
                    [
                        p["catid1_dict"]["meanproductgsd"],
                        p["catid2_dict"]["meanproductgsd"],
                    ]
                )
            except Exception:
                logger.warning(
                    "\n\nCould not read stereopair metadata to get mean GSD. Residuals will remain in pixels.\n\n"
                )
                mean_gsd = 1.0

            resid_gdf["mean_residual_meters"] = (
                resid_gdf["mean_residual"].values * mean_gsd
            )

        resid_gdf.filename = os.path.basename(csv_path)
        return resid_gdf

    def get_geodiff_gdf(self, csv_path):
        """
        Read geodiff output into a GeoDataFrame.

        Reads a geodiff output CSV file and converts it to a GeoDataFrame
        with appropriate geometry.

        Parameters
        ----------
        csv_path : str
            Path to the geodiff CSV file

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing geodiff data with point geometry

        Notes
        -----
        The returned GeoDataFrame has a 'filename' attribute set to the
        base name of the input file. The main data column is 'height_diff_meters',
        which contains the height differences between residual points and the
        reference DEM.
        """
        cols = [
            "lon",
            "lat",
            "height_diff_meters",
        ]

        geodiff_df = pd.read_csv(csv_path, skiprows=7, names=cols)

        geodiff_gdf = gpd.GeoDataFrame(
            geodiff_df,
            geometry=gpd.points_from_xy(
                geodiff_df["lon"], geodiff_df["lat"], crs="EPSG:4326"
            ),
        )

        geodiff_gdf.filename = os.path.basename(csv_path)
        return geodiff_gdf

    def get_initial_final_residuals_gdfs(self, residuals_in_meters=True):
        """
        Get both initial and final residuals as GeoDataFrames.

        A convenience method that combines get_csv_paths() and
        get_residuals_gdf() to obtain both initial and final
        bundle adjustment residuals.

        Parameters
        ----------
        residuals_in_meters : bool, optional
            Whether to convert residuals from pixels to meters using
            the mean GSD from the stereopair metadata, default is True

        Returns
        -------
        tuple of geopandas.GeoDataFrame
            GeoDataFrames containing initial and final residual data

        Notes
        -----
        This is a convenience method that calls get_csv_paths() and
        get_residuals_gdf() for both initial and final residual files.
        """
        resid_initial_path, resid_final_path = self.get_csv_paths()
        resid_initial_gdf = self.get_residuals_gdf(
            resid_initial_path, residuals_in_meters
        )
        resid_final_gdf = self.get_residuals_gdf(resid_final_path, residuals_in_meters)
        return resid_initial_gdf, resid_final_gdf

    def get_initial_final_geodiff_gdfs(self):
        """
        Get both initial and final geodiff results as GeoDataFrames.

        A convenience method that combines get_csv_paths() and
        get_geodiff_gdf() to obtain both initial and final
        geodiff results.

        Returns
        -------
        tuple of geopandas.GeoDataFrame
            GeoDataFrames containing initial and final geodiff data

        Notes
        -----
        This is a convenience method that calls get_csv_paths(geodiff_files=True)
        and get_geodiff_gdf() for both initial and final geodiff files. If the
        geodiff files don't exist, they will be generated using generate_geodiff().
        """
        geodiff_initial_path, geodiff_final_path = self.get_csv_paths(
            geodiff_files=True
        )
        geodiff_initial_gdf = self.get_geodiff_gdf(geodiff_initial_path)
        geodiff_final_gdf = self.get_geodiff_gdf(geodiff_final_path)
        return geodiff_initial_gdf, geodiff_final_gdf

    def get_mapproj_residuals_gdf(self):
        """
        Get map-projected residuals as a GeoDataFrame.

        Reads the map-projected match offsets file produced by
        bundle_adjust and converts it to a GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame containing map-projected residual data

        Raises
        ------
        ValueError
            If the map-projected match offsets file cannot be found

        Notes
        -----
        The main data column is 'mapproj_ip_dist_meters', which contains
        the distance between matched interest points in the map-projected
        images after bundle adjustment.
        """
        path = glob_file(self.full_directory, "*-mapproj_match_offsets.txt")
        if path is None:
            raise ValueError("\n\nMapProj Residuals TXT file not found.\n\n")

        cols = ["lon", "lat", "height_above_datum", "mapproj_ip_dist_meters"]
        resid_mapprojected_df = pd.read_csv(path, skiprows=2, names=cols)
        resid_mapprojected_gdf = gpd.GeoDataFrame(
            resid_mapprojected_df,
            geometry=gpd.points_from_xy(
                resid_mapprojected_df["lon"],
                resid_mapprojected_df["lat"],
                crs="EPSG:4326",
            ),
        )
        return resid_mapprojected_gdf

    def get_propagated_triangulation_uncert_df(self):
        """
        Get propagated triangulation uncertainty as a DataFrame.

        Reads the triangulation uncertainty file produced by
        bundle_adjust, which contains statistics on the expected
        horizontal and vertical error in the DEM.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing triangulation uncertainty data

        Raises
        ------
        ValueError
            If the triangulation uncertainty file cannot be found

        Notes
        -----
        The triangulation uncertainty file contains statistics on how
        the bundle adjustment residuals propagate to uncertainty in
        triangulation. This is an estimate of the final DEM error.
        """
        path = glob_file(self.full_directory, "*-triangulation_uncertainty.txt")
        if path is None:
            raise ValueError("\n\nTriangulation Uncertainty TXT file not found.\n\n")

        cols = [
            "left_image",
            "right_image",
            "horiz_error_median",
            "vert_error_median",
            "horiz_error_mean",
            "vert_error_mean",
            "horiz_error_stddev",
            "vert_error_stddev",
            "num_meas",
        ]
        resid_triangulation_uncert_df = pd.read_csv(
            path, sep=" ", skiprows=2, names=cols
        )
        return resid_triangulation_uncert_df


class PlotBundleAdjustFiles(Plotter):
    """
    Plot bundle adjustment results from GeoDataFrames.

    This class extends the base Plotter class to provide specialized
    plotting functionality for bundle adjustment results. It can create
    visualizations of residuals, geodiff results, and other bundle
    adjustment outputs.

    Attributes
    ----------
    geodataframes : list of geopandas.GeoDataFrame
        List of GeoDataFrames containing bundle adjustment data to plot
    title : str
        Plot title, inherited from Plotter class

    Examples
    --------
    >>> ba_reader = ReadBundleAdjustFiles('/path/to/asp', 'ba')
    >>> initial_gdf, final_gdf = ba_reader.get_initial_final_residuals_gdfs()
    >>> ba_plotter = PlotBundleAdjustFiles([initial_gdf, final_gdf], title="Bundle Adjustment Residuals")
    >>> ba_plotter.plot_n_gdfs(column_name="mean_residual_meters", cbar_label="Mean residual (m)")
    """

    def __init__(self, geodataframes, **kwargs):
        """
        Initialize the PlotBundleAdjustFiles object.

        Parameters
        ----------
        geodataframes : list of geopandas.GeoDataFrame
            List of GeoDataFrames containing bundle adjustment data to plot
        **kwargs : dict, optional
            Additional keyword arguments to pass to the Plotter base class

        Raises
        ------
        ValueError
            If geodataframes is not a list
        """
        super().__init__(**kwargs)
        if not isinstance(geodataframes, list):
            raise ValueError("\n\nInput must be a list of GeoDataFrames\n\n")
        self.geodataframes = geodataframes

    def gdf_percentile_stats(self, gdf, column_name="mean_residual"):
        """
        Calculate percentile statistics for a GeoDataFrame column.

        Computes the 25th, 50th, 84th, and 95th percentiles for a
        specified column in a GeoDataFrame.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            GeoDataFrame to analyze
        column_name : str, optional
            Column to calculate statistics for, default is "mean_residual"

        Returns
        -------
        list
            List of statistics at the 25th, 50th, 84th, and 95th percentiles

        Notes
        -----
        These percentiles are commonly used in remote sensing and
        photogrammetry to assess error distributions. The 50th percentile
        is the median, while the 84th percentile is approximately
        equivalent to one standard deviation in a normal distribution.
        """
        stats = gdf[column_name].quantile([0.25, 0.50, 0.84, 0.95]).round(2).tolist()
        return stats

    def plot_n_gdfs(
        self,
        column_name="mean_residual",
        cbar_label="Mean residual (px)",
        clip_final=True,
        clim=None,
        common_clim=True,
        symm_clim=False,
        cmap="inferno",
        map_crs="EPSG:4326",
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        """
        Plot multiple GeoDataFrames in a grid layout.

        Creates a figure with multiple subplots, one for each GeoDataFrame
        in the geodataframes list. Each subplot shows the spatial distribution
        of the specified column values.

        Parameters
        ----------
        column_name : str, optional
            Column to visualize, default is "mean_residual"
        cbar_label : str, optional
            Label for the colorbar, default is "Mean residual (px)"
        clip_final : bool, optional
            Whether to clip the final plot to prevent autoscaling,
            default is True
        clim : tuple or None, optional
            Color limits as (min, max), default is None (auto)
        common_clim : bool, optional
            Whether to use the same color limits for all plots,
            default is True
        symm_clim : bool, optional
            Whether to use symmetric color limits, default is False
        cmap : str, optional
            Matplotlib colormap name, default is "inferno"
        map_crs : str, optional
            Coordinate reference system for plotting, default is "EPSG:4326"
        save_dir : str or None, optional
            Directory to save the figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for the saved figure, default is None
        **ctx_kwargs : dict, optional
            Additional keyword arguments for contextily basemap

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        Each subplot includes a text box showing statistics for the
        displayed data, including the number of points and percentile
        values. The layout automatically adjusts based on the number
        of GeoDataFrames to plot.
        """

        # Get rows and columns and create subplots
        n = len(self.geodataframes)
        nrows = (n + 3) // 4
        ncols = min(n, 4)
        if n == 1:
            fig, axa = plt.subplots(1, 1, figsize=(8, 6))
            axa = [axa]
        else:
            fig, axa = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True
            )
            axa = axa.flatten()

        # Plot each GeoDataFrame
        for i, gdf in enumerate(self.geodataframes):
            gdf = gdf.sort_values(by=column_name).to_crs(map_crs)

            if clim is None:
                clim = ColorBar(symm=symm_clim).get_clim(gdf[column_name])

            if common_clim:
                self.plot_geodataframe(
                    ax=axa[i],
                    gdf=gdf,
                    clim=clim,
                    column_name=column_name,
                    cbar_label=cbar_label,
                    cmap=cmap,
                    **ctx_kwargs,
                )
            else:
                self.plot_geodataframe(
                    ax=axa[i],
                    gdf=gdf,
                    column_name=column_name,
                    cbar_label=cbar_label,
                    cmap=cmap,
                    **ctx_kwargs,
                )

            if clip_final and i == n - 1:
                axa[i].autoscale(False)

            # Show some statistics and information
            stats = self.gdf_percentile_stats(gdf, column_name)
            stats_text = f"(n={gdf.shape[0]})\n" + "\n".join(
                f"{quantile*100:.0f}th: {stat}"
                for quantile, stat in zip([0.25, 0.50, 0.84, 0.95], stats)
            )
            axa[i].text(
                0.05,
                0.95,
                stats_text,
                transform=axa[i].transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Clean up axes and tighten layout
        for i in range(n, nrows * ncols):
            fig.delaxes(axa[i])
        fig.suptitle(self.title, size=10)
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        self.save(fig, save_dir, fig_fn)


def _enu_basis(center_ecef):
    """
    Build a local East-North-Up basis at an ECEF point.

    Parameters
    ----------
    center_ecef : array-like
        A single ECEF (EPSG:4978) coordinate as [x, y, z] in meters.

    Returns
    -------
    tuple of numpy.ndarray
        The (east, north, up) unit vectors expressed in ECEF, so that the
        east/north/up component of any ECEF vector ``v`` is ``v @ east`` etc.

    Notes
    -----
    The basis is computed from the geodetic latitude and longitude of the
    point (the sub-satellite location for a camera center). This lets us
    decompose the ECEF camera-center shift from a ``.adjust`` file into the
    horizontal and vertical components that ASP reports in
    ``camera_offsets.txt``.
    """
    lat, lon, _ = Transformer.from_crs("EPSG:4978", "EPSG:4326").transform(*center_ecef)
    lat, lon = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(up, east)
    return east, north, up


def _camera_label(path):
    """
    Human-readable label for a camera, from its filename.

    Strips only the deterministic camera-file extension (``.adjusted_state.json``,
    ``.adjusted_state.adjust``, ``.adjust``, or ``.xml``) -- it makes no
    assumptions about the ASP output prefix (the ``-o`` value, e.g. ``run`` or
    ``ba_mvs_csm``) or ``_corr``-style suffixes, so the label is just the real
    filename stem. This is display-only and never used to associate files.
    """
    name = os.path.basename(path)
    for ext in (".adjusted_state.json", ".adjusted_state.adjust", ".adjust", ".xml"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def _camera_center_from_xml(xml_path):
    """
    Return the image-center ECEF camera center from a DigitalGlobe XML.

    Reads the ``<EPHEMLIST>`` satellite ephemeris (ECF positions, meters) and
    interpolates the position at the time the sensor imaged the center image
    line -- the sub-satellite point at mid-acquisition, used to place the camera
    on the map and build its local ENU frame. This is the original
    (pre-adjustment) camera center: DG ``bundle_adjust`` runs store the
    optimization only as a ``.adjust`` delta, with no ``.adjusted_state.json``.

    The center-line time is ``FIRSTLINETIME + (NUMROWS / 2) / AVGLINERATE``,
    converted to an ephemeris sample index via ``STARTTIME`` / ``TIMEINTERVAL``.
    Falls back to the ephemeris mean if any of those tags are missing.

    Parameters
    ----------
    xml_path : str
        Path to a DigitalGlobe image metadata ``.xml`` file.

    Returns
    -------
    numpy.ndarray
        Length-3 ECEF (EPSG:4978) coordinate in meters.
    """
    ephem = np.array(
        [row.split() for row in get_xml_tag(xml_path, "EPHEMLIST", all=True)],
        dtype=np.float64,
    )
    # Columns are: point_num, X, Y, Z, dX, dY, dZ, covariance...
    positions = ephem[:, 1:4]
    try:
        start = pd.to_datetime(get_xml_tag(xml_path, "STARTTIME"))
        dt = float(get_xml_tag(xml_path, "TIMEINTERVAL"))
        first_line = pd.to_datetime(get_xml_tag(xml_path, "FIRSTLINETIME"))
        num_rows = float(get_xml_tag(xml_path, "NUMROWS"))
        line_rate = float(get_xml_tag(xml_path, "AVGLINERATE"))
        center_time = first_line + pd.Timedelta((num_rows / 2.0) / line_rate, unit="s")
        index = (center_time - start).total_seconds() / dt
        sample = np.arange(len(positions))
        return np.array([np.interp(index, sample, positions[:, k]) for k in range(3)])
    except Exception:
        return positions.mean(axis=0)


class ReadBundleAdjustCameras:
    """
    Read before/after camera geometry from an ASP bundle_adjust folder.

    Unlike :class:`asp_plot.csm_camera.csm_camera_summary_plot`, which requires
    the user to supply the original (pre-adjustment) camera files, this reader
    works directly on a ``bundle_adjust`` output directory. It combines three
    self-contained products that ASP always writes there:

    - ``*.adjust`` -- the rigid adjustment (ECEF translation + rotation
      quaternion) applied to each camera. Per ASP's convention a world point
      projects the same in the original camera as ``R * (P - C) + C + T`` in
      the adjusted camera, so the translation ``T`` is the bulk camera-center
      shift (exact at the camera center for pixel (0, 0)).
    - ``*.adjusted_state.json`` -- the optimized CSM camera state, used to
      locate each camera center in space.
    - ``*camera_offsets.txt`` (optional) -- ASP's authoritative per-camera
      horizontal and vertical camera-center change, in the local North-East-Down
      frame. When present it is used for the reported magnitudes (it also folds
      in the rotation lever-arm that the bulk translation ``T`` does not).

    Parameters
    ----------
    directory : str
        Root directory of ASP processing.
    bundle_adjust_directory : str
        Subdirectory containing bundle adjustment outputs.

    Examples
    --------
    >>> reader = ReadBundleAdjustCameras('/path/to/asp', 'ba')
    >>> gdf = reader.get_camera_optimization_gdf(map_crs=32616)
    """

    def __init__(self, directory, bundle_adjust_directory):
        self.directory = os.path.expanduser(directory)
        self.bundle_adjust_directory = bundle_adjust_directory
        self.full_directory = os.path.join(self.directory, bundle_adjust_directory)

    def get_camera_offsets_df(self):
        """
        Read ``*camera_offsets.txt`` into a DataFrame if it exists.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with columns ``image``, ``horizontal_offset_m``, and
            ``vertical_offset_m`` (one row per input image, in ASP's order).
            Returns None if the file is not present (it is only written by
            recent ASP versions).
        """
        # camera_offsets.txt is optional (only written by recent ASP versions),
        # so look it up directly to avoid glob_file's "missing" warning.
        matches = glob.glob(os.path.join(self.full_directory, "*camera_offsets.txt"))
        if not matches:
            return None
        return pd.read_csv(
            matches[0],
            sep=r"\s+",
            comment="#",
            header=None,
            names=["image", "horizontal_offset_m", "vertical_offset_m"],
        )

    @staticmethod
    def _read_list_file(path):
        """Read an ASP ``*_list.txt`` file as ordered, comment-free lines."""
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    entries.append(line)
        return entries

    def _offsets_by_camera_basename(self):
        """
        Associate ``camera_offsets.txt`` rows with cameras via ``camera_list.txt``.

        ASP writes ``camera_offsets.txt`` (one row per input image) and
        ``camera_list.txt`` (one output camera per line) together and **in the
        same input order**. Zipping them by position is exact and needs no
        filename-string assumptions (no ``run-``/``_corr`` guessing). The result
        is keyed by the camera file's basename so a discovered camera can be
        looked up directly.

        Returns
        -------
        dict or None
            Maps each camera basename (e.g. ``run-<id>.adjusted_state.json`` or
            ``<id>.xml``) to ``(horizontal_offset_m, vertical_offset_m)``. Returns
            None if either companion file is absent or their lengths disagree.
        """
        offsets_df = self.get_camera_offsets_df()
        list_matches = glob.glob(os.path.join(self.full_directory, "*camera_list.txt"))
        if offsets_df is None or not list_matches:
            return None
        cameras = [
            os.path.basename(entry) for entry in self._read_list_file(list_matches[0])
        ]
        if len(cameras) != len(offsets_df):
            logger.warning(
                "\n\ncamera_list.txt and camera_offsets.txt lengths differ "
                f"({len(cameras)} vs {len(offsets_df)}); ignoring ASP offsets and "
                "falling back to the .adjust translation.\n\n"
            )
            return None
        return {
            camera: (float(row.horizontal_offset_m), float(row.vertical_offset_m))
            for camera, (_, row) in zip(cameras, offsets_df.iterrows())
        }

    def _find_state_file(self, adjust_path):
        """
        Find the ``.adjusted_state.json`` matching a ``.adjust`` file, if any.

        Handles both naming conventions seen in ASP output: ``<base>.adjust``
        alongside ``<base>.adjusted_state.json`` (WorldView/CSM runs) and
        ``<base>.adjusted_state.adjust`` alongside ``<base>.adjusted_state.json``
        (e.g. ASTER jitter runs). Returns None for DigitalGlobe runs, which
        write only the ``.adjust`` delta with no state file.
        """
        base = adjust_path[: -len(".adjust")]
        candidates = [
            base + ".adjusted_state.json",  # <base>.adjust
            base + ".json",  # <base>.adjusted_state.adjust
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _index_original_xmls(self, original_cameras_directory=None):
        """
        Build a ``{filename stem: path}`` index of original camera XMLs.

        Used to locate the pre-adjustment camera center for DigitalGlobe runs
        (which have no ``.adjusted_state.json``). When
        ``original_cameras_directory`` is given, only that directory is searched;
        otherwise the bundle_adjust directory and its parent are searched (ASP
        typically writes the ``ba_*`` output as a subdirectory of the folder
        holding the input ``.xml`` cameras).

        Parameters
        ----------
        original_cameras_directory : str or None, optional
            Directory holding the original ``.xml`` cameras. If None, auto-search
            the BA directory and its parent.

        Returns
        -------
        dict
            Maps each XML filename stem (basename without ``.xml``) to its path.
        """
        if original_cameras_directory:
            search_dirs = [os.path.expanduser(original_cameras_directory)]
        else:
            search_dirs = [
                self.full_directory,
                os.path.dirname(self.full_directory.rstrip(os.sep)),
            ]
        index = {}
        for directory in search_dirs:
            for path in glob.glob(os.path.join(directory, "*.xml")):
                stem = os.path.basename(path)[: -len(".xml")]
                index.setdefault(stem, path)
        return index

    @staticmethod
    def _match_original_xml(adjust_path, xml_index):
        """
        Match a ``.adjust`` file to an original camera XML by filename.

        ASP names DG adjustments ``<prefix>-<camera>.adjust`` where ``<camera>``
        is the original XML's basename stem (e.g. a CATID like
        ``10300100D044F700.r100``). The longest XML stem that appears in the
        ``.adjust`` filename is taken as the match, so distinct CATIDs never
        cross-match.
        """
        adjust_base = os.path.basename(adjust_path)
        best = None
        for stem, path in xml_index.items():
            if stem in adjust_base and (best is None or len(stem) > len(best[0])):
                best = (stem, path)
        return best[1] if best else None

    @staticmethod
    def read_adjust_file(adjust_path):
        """
        Parse an ASP ``.adjust`` file.

        Parameters
        ----------
        adjust_path : str
            Path to a ``.adjust`` file.

        Returns
        -------
        tuple
            ``(translation, rotation)`` where ``translation`` is a length-3
            numpy array of the ECEF camera-center shift in meters and
            ``rotation`` is a :class:`scipy.spatial.transform.Rotation`.

        Notes
        -----
        The first line holds the translation ``x y z`` (meters); the second
        holds the rotation quaternion in ASP's ``w x y z`` order, which is
        reordered to the ``x y z w`` order that SciPy expects.
        """
        with open(adjust_path, "r") as f:
            lines = f.read().split("\n")
        translation = np.array([float(x) for x in lines[0].split()])
        w, x, y, z = (float(v) for v in lines[1].split())
        rotation = Rotation.from_quat([x, y, z, w])
        return translation, rotation

    def _representative_center(self, state_path):
        """
        Return the image-center ECEF camera center for a camera state file.

        For frame cameras this is the single camera center. For linescan
        cameras it is the trajectory position at the center image line (the
        sub-satellite point at mid-acquisition), which is more meaningful than
        the trajectory mean when the stored ephemeris is padded beyond the image
        acquisition window. Falls back to the trajectory mean if the center-line
        time cannot be computed.
        """
        positions, _ = read_positions_rotations_from_file(state_path)
        positions = np.array(positions)
        if len(positions) < 2 or not isLinescan(state_path):
            return positions.mean(axis=0)
        try:
            j = read_csm_cam(state_path)
            center_time = getTimeAtLine(j, (j["m_nLines"] - 1) / 2.0)
            times = j["m_t0Ephem"] + j["m_dtEphem"] * np.arange(len(positions))
            return np.array(
                [np.interp(center_time, times, positions[:, k]) for k in range(3)]
            )
        except Exception:
            return positions.mean(axis=0)

    def get_camera_optimization_gdf(
        self, map_crs=None, original_cameras_directory=None
    ):
        """
        Build a per-camera GeoDataFrame of before/after camera changes.

        Discovery is driven by the ``.adjust`` files, which ASP writes for every
        camera. Each camera's absolute center comes from its
        ``.adjusted_state.json`` (WorldView/CSM and jitter runs) or, when that is
        absent (DigitalGlobe runs), from the original camera ``.xml`` ephemeris.

        Parameters
        ----------
        map_crs : int or None, optional
            EPSG code (e.g. a UTM zone) for the output geometry. If None, the
            geometry is returned in geographic coordinates (EPSG:4326). A
            projected CRS is recommended for the position quiver so the
            east/north shift components align with the map axes.
        original_cameras_directory : str or None, optional
            Directory holding the original ``.xml`` cameras, used only for
            DigitalGlobe runs that lack ``.adjusted_state.json``. If None, the
            BA directory and its parent are searched automatically.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per camera, with geometry at the (projected) camera center
            and columns:

            - ``camera_id`` : camera filename stem (display label).
            - ``t_east``, ``t_north``, ``t_up`` : ECEF translation ``T``
              decomposed into the local ENU frame (meters).
            - ``t_horizontal`` : horizontal magnitude of ``T`` (meters).
            - ``adj_roll``, ``adj_pitch``, ``adj_yaw`` : the ``.adjust``
              rotation as intrinsic XYZ Euler angles (degrees).
            - ``horizontal_offset_m``, ``vertical_offset_m`` : ASP's reported
              camera-center change from ``camera_offsets.txt`` when available,
              otherwise filled from ``t_horizontal`` / ``t_up``.
            - ``offsets_from_asp`` : True if the offsets came from
              ``camera_offsets.txt``, False if derived from ``T``.

        Raises
        ------
        ValueError
            If no ``.adjust`` files are found in the directory.
        """
        adjust_paths = glob_file(self.full_directory, "*.adjust", all_files=True)
        if adjust_paths is None:
            raise ValueError(
                "\n\nNo *.adjust files found. This reader needs the per-camera "
                ".adjust files written by bundle_adjust (or jitter_solve).\n\n"
            )

        offsets_by_camera = self._offsets_by_camera_basename()
        xml_index = None  # built lazily, only if a DG (state-less) camera appears

        rows, centers_ecef = [], []
        for adjust_path in sorted(adjust_paths):
            state_path = self._find_state_file(adjust_path)

            # DigitalGlobe runs have no state file; fall back to the original
            # camera XML for the absolute (pre-adjustment) center.
            xml_path = None
            if state_path is None:
                if xml_index is None:
                    xml_index = self._index_original_xmls(original_cameras_directory)
                xml_path = self._match_original_xml(adjust_path, xml_index)
                if xml_path is None:
                    logger.warning(
                        f"\n\nNo .adjusted_state.json or matching original .xml "
                        f"camera for {adjust_path}. Skipping. (For DigitalGlobe "
                        "runs, point --original_cameras_directory at the input "
                        ".xml cameras.)\n\n"
                    )
                    continue

            # A corrupt/truncated .adjust, state, or XML file should not sink the
            # whole run; skip that one camera with a warning.
            camera_path = state_path if state_path is not None else xml_path
            try:
                translation, rotation = self.read_adjust_file(adjust_path)
                if state_path is not None:
                    center_ecef = self._representative_center(state_path)
                else:
                    center_ecef = _camera_center_from_xml(xml_path)
                camera_id = _camera_label(camera_path)
            except Exception as e:
                logger.warning(
                    f"\n\nCould not parse camera files for {adjust_path} "
                    f"({type(e).__name__}: {e}). Skipping.\n\n"
                )
                continue
            east, north, up = _enu_basis(center_ecef)
            t_east, t_north, t_up = (
                float(translation @ east),
                float(translation @ north),
                float(translation @ up),
            )
            roll, pitch, yaw = rotation.as_euler("XYZ", degrees=True)

            row = {
                "camera_id": camera_id,
                "t_east": t_east,
                "t_north": t_north,
                "t_up": t_up,
                "t_horizontal": float(np.hypot(t_east, t_north)),
                "adj_roll": float(roll),
                "adj_pitch": float(pitch),
                "adj_yaw": float(yaw),
            }

            offset = None
            if offsets_by_camera is not None:
                offset = offsets_by_camera.get(os.path.basename(camera_path))

            if offset is not None:
                row["horizontal_offset_m"], row["vertical_offset_m"] = offset
                row["offsets_from_asp"] = True
            else:
                row["horizontal_offset_m"] = row["t_horizontal"]
                row["vertical_offset_m"] = abs(t_up)
                row["offsets_from_asp"] = False

            rows.append(row)
            centers_ecef.append(center_ecef)

        if not rows:
            raise ValueError(
                "\n\nFound *.adjust files but no camera could be built (no matching "
                ".adjusted_state.json or original .xml cameras, or unparseable files). "
                "Cannot build the camera optimization GeoDataFrame.\n\n"
            )

        gdf = gpd.GeoDataFrame(
            pd.DataFrame(rows),
            geometry=[Point(*c) for c in centers_ecef],
            crs="EPSG:4978",
        )
        gdf = gdf.to_crs(epsg=map_crs) if map_crs else gdf.to_crs(epsg=4326)
        return gdf


class PlotBundleAdjustCameras(Plotter):
    """
    Visualize before/after camera position and orientation changes.

    Consumes the GeoDataFrame from
    :meth:`ReadBundleAdjustCameras.get_camera_optimization_gdf` and renders the
    three complementary views requested in issues #95 and #43:

    1. A map-view quiver of the horizontal camera-center shift, with the
       vertical change encoded as a diverging (RdBu) color.
    2. Per-camera bars of the horizontal and vertical camera-center change.
    3. A quiver of the orientation change (yaw/roll as the arrow direction,
       pitch as color).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of
        :meth:`ReadBundleAdjustCameras.get_camera_optimization_gdf`.
    **kwargs
        Forwarded to :class:`asp_plot.utils.Plotter`.
    """

    def __init__(self, gdf, **kwargs):
        super().__init__(**kwargs)
        self.gdf = gdf

    def plot_position_change_quiver(
        self, ax=None, cmap="RdBu", arrow_frac=0.18, save_dir=None, fig_fn=None
    ):
        """
        Map-view quiver of the horizontal camera-center shift.

        Each arrow points in the direction of the horizontal camera-center
        translation (``t_east``, ``t_north``); its color encodes the vertical
        change (``t_up``) on a symmetric diverging scale. A reference arrow
        (quiverkey) shows the true scale in meters.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on. A new figure is created if None.
        cmap : str, optional
            Diverging colormap for the vertical change, default "RdBu".
        arrow_frac : float, optional
            Target length of the largest arrow as a fraction of the map width,
            default 0.18.
        save_dir, fig_fn : str or None, optional
            If both are given (and a new figure was created), save the figure.
        """
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 7))

        gdf = self.gdf
        x, y = gdf.geometry.x.values, gdf.geometry.y.values
        vmax = max(float(np.abs(gdf.t_up).max()), 1e-9)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        max_h = max(float(gdf.t_horizontal.max()), 1e-9)
        span = max(x.max() - x.min(), y.max() - y.min(), 1.0)
        scale = max_h / (arrow_frac * span)

        q = ax.quiver(
            x,
            y,
            gdf.t_east,
            gdf.t_north,
            gdf.t_up,
            cmap=cmap,
            norm=norm,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.012,
            edgecolor="k",
            linewidth=0.4,
        )
        ax.quiverkey(
            q,
            0.80,
            0.06,
            max_h,
            f"{max_h:.2f} m",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": 8},
        )
        ax.scatter(x, y, c="k", s=6, zorder=3)
        cbar = plt.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, extend="both", pad=0.02
        )
        cbar.set_label("Vertical change (m, + up)", fontsize=9)
        ax.set_title("Camera position change (before → after)", fontsize=11)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.margins(0.25)

        if created:
            self.save(fig, save_dir, fig_fn)
        return ax

    def plot_center_offset_bars(self, ax=None, save_dir=None, fig_fn=None):
        """
        Per-camera bars of horizontal and vertical camera-center change.

        Uses ASP's ``camera_offsets.txt`` values when available (see
        ``offsets_from_asp``); otherwise falls back to the horizontal/vertical
        components of the ``.adjust`` translation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on. A new figure is created if None.
        save_dir, fig_fn : str or None, optional
            If both are given (and a new figure was created), save the figure.
        """
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 5))

        gdf = self.gdf
        ids = gdf.camera_id.values
        xi = np.arange(len(ids))
        ax.bar(
            xi - 0.2,
            gdf.horizontal_offset_m,
            0.4,
            label="Horizontal",
            color="#4169E1",
        )
        ax.bar(
            xi + 0.2,
            gdf.vertical_offset_m.abs(),
            0.4,
            label="Vertical",
            color="#87CEEB",
        )
        ax.set_xticks(xi)
        ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Camera-center change (m)", fontsize=9)
        source = (
            "camera_offsets.txt"
            if bool(gdf.offsets_from_asp.any())
            else "|.adjust translation|"
        )
        ax.set_title(f"Per-camera center displacement\n(from {source})", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.8)

        if created:
            self.save(fig, save_dir, fig_fn)
        return ax

    def plot_orientation_change_quiver(
        self, ax=None, cmap="PuOr", save_dir=None, fig_fn=None
    ):
        """
        Quiver of the per-camera orientation change from the ``.adjust`` file.

        Following issue #43, the arrow direction encodes yaw (x) and roll (y)
        change and the color encodes pitch change, all in degrees.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axes to draw on. A new figure is created if None.
        cmap : str, optional
            Diverging colormap for the pitch change, default "PuOr".
        save_dir, fig_fn : str or None, optional
            If both are given (and a new figure was created), save the figure.
        """
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(8, 5))

        gdf = self.gdf
        ids = gdf.camera_id.values
        xi = np.arange(len(ids))
        pmax = max(float(np.abs(gdf.adj_pitch).max()), 1e-12)
        norm = TwoSlopeNorm(vmin=-pmax, vcenter=0.0, vmax=pmax)
        q = ax.quiver(
            xi,
            np.zeros(len(xi)),
            gdf.adj_yaw,
            gdf.adj_roll,
            gdf.adj_pitch,
            cmap=cmap,
            norm=norm,
            angles="xy",
        )
        ax.quiverkey(
            q,
            0.72,
            0.06,
            float(np.abs(np.hypot(gdf.adj_yaw, gdf.adj_roll)).max()) or 1e-6,
            "roll/yaw change",
            labelpos="E",
            coordinates="axes",
            fontproperties={"size": 8},
        )
        cbar = plt.colorbar(
            ScalarMappable(norm=norm, cmap=cmap), ax=ax, extend="both", pad=0.02
        )
        cbar.set_label("Pitch change (°)", fontsize=9)
        ax.set_xticks(xi)
        ax.set_xticklabels(ids, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Roll change (°)", fontsize=9)
        ax.set_title(
            "Orientation change\n(arrow: yaw = x, roll = y; color: pitch)", fontsize=11
        )
        ax.axhline(0, color="k", lw=0.5)
        ax.margins(0.25)

        if created:
            self.save(fig, save_dir, fig_fn)
        return ax

    def summary_plot(self, save_dir=None, fig_fn=None):
        """
        Combined three-panel summary of camera position and orientation change.

        Draws, left to right, the position-change map quiver, the per-camera
        center-displacement bars, and the orientation-change quiver.

        Parameters
        ----------
        save_dir, fig_fn : str or None, optional
            If both are given, save the figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=(19, 6))
        self.plot_position_change_quiver(ax=axes[0])
        self.plot_center_offset_bars(ax=axes[1])
        self.plot_orientation_change_quiver(ax=axes[2])
        if self.title:
            fig.suptitle(self.title, size=12)
        self.save(fig, save_dir, fig_fn)
        return fig
