import os
import glob
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import contextily as ctx
from asp_plot.utils import ColorBar, Plotter, save_figure


class ReadBundleAdjustFiles:
    def __init__(self, directory, bundle_adjust_directory):
        self.directory = directory
        self.bundle_adjust_directory = bundle_adjust_directory

    def get_csv_paths(self, geodiff_files=False):
        filenames = [
            "*-initial_residuals_pointmap.csv",
            "*-final_residuals_pointmap.csv",
        ]

        if geodiff_files:
            filenames = [f.replace(".csv", "-diff.csv") for f in filenames]

        paths = [
            glob.glob(os.path.join(self.directory, self.bundle_adjust_directory, f))[0]
            for f in filenames
        ]

        for path in paths:
            if not os.path.isfile(path):
                raise ValueError(f"CSV file not found: {path}")

        initial, final = paths
        return initial, final

    def get_residuals_gdf(self, csv_path):
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

        resid_gdf.filename = os.path.basename(csv_path)
        return resid_gdf

    def get_geodiff_gdf(self, csv_path):
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

    def get_initial_final_residuals_gdfs(self):
        resid_initial_path, resid_final_path = self.get_csv_paths()
        resid_initial_gdf = self.get_residuals_gdf(resid_initial_path)
        resid_final_gdf = self.get_residuals_gdf(resid_final_path)
        return resid_initial_gdf, resid_final_gdf

    def get_initial_final_geodiff_gdfs(self):
        geodiff_initial_path, geodiff_final_path = self.get_csv_paths(
            geodiff_files=True
        )
        geodiff_initial_gdf = self.get_geodiff_gdf(geodiff_initial_path)
        geodiff_final_gdf = self.get_geodiff_gdf(geodiff_final_path)
        return geodiff_initial_gdf, geodiff_final_gdf

    def get_mapproj_residuals_gdf(self):
        path = glob.glob(
            os.path.join(
                self.directory,
                self.bundle_adjust_directory,
                "*-mapproj_match_offsets.txt",
            )
        )[0]
        if not os.path.isfile(path):
            raise ValueError(f"MapProj Residuals TXT file not found: {path}")

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
        path = glob.glob(
            os.path.join(
                self.directory,
                self.bundle_adjust_directory,
                "*-triangulation_uncertainty.txt",
            )
        )[0]
        if not os.path.isfile(path):
            raise ValueError(f"Triangulation Uncertainty TXT file not found: {path}")

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
    def __init__(self, geodataframes, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(geodataframes, list):
            raise ValueError("Input must be a list of GeoDataFrames")
        self.geodataframes = geodataframes

    def gdf_percentile_stats(self, gdf, column_name="mean_residual"):
        stats = gdf[column_name].quantile([0.25, 0.50, 0.84, 0.95]).round(2).tolist()
        return stats

    def plot_n_gdfs(
        self,
        column_name="mean_residual",
        cbar_label="Mean Residual (m)",
        clip_final=True,
        clim=None,
        common_clim=True,
        cmap="inferno",
        map_crs="EPSG:4326",
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):

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
                clim = ColorBar().get_clim(gdf[column_name])

            if common_clim:
                self.plot_geodataframe(
                    ax=axa[i],
                    gdf=gdf,
                    clim=clim,
                    column_name=column_name,
                    cbar_label=cbar_label,
                    cmap=cmap,
                )
            else:
                self.plot_geodataframe(
                    ax=axa[i],
                    gdf=gdf,
                    column_name=column_name,
                    cbar_label=cbar_label,
                    cmap=cmap,
                )

            ctx.add_basemap(ax=axa[i], **ctx_kwargs)

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
