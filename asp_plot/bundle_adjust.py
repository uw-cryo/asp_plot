import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import (
    ColorBar,
    Plotter,
    glob_file,
    run_subprocess_command,
    save_figure,
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
        self.directory = directory
        self.bundle_adjust_directory = bundle_adjust_directory
        self.full_directory = os.path.join(directory, bundle_adjust_directory)

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
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
