import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import contextily as ctx
from asp_plot.utils import ColorBar, Plotter


class ReadResiduals:
    def __init__(self, directory, ba_prefix):
        self.directory = directory
        self.ba_prefix = ba_prefix

    def get_residual_csv_paths(self):
        filenames = [
            f"{self.ba_prefix}-initial_residuals_pointmap.csv",
            f"{self.ba_prefix}-final_residuals_pointmap.csv",
        ]

        paths = [
            os.path.join(os.path.expanduser(self.directory), filename)
            for filename in filenames
        ]

        for path in paths:
            if not os.path.isfile(path):
                raise ValueError(f"Residuals CSV file not found: {path}")

        init_resid_path, final_resid_path = paths

        return init_resid_path, final_resid_path

    def read_residuals_csv(self, csv_path):
        resid_cols = [
            "lon",
            "lat",
            "height_above_datum",
            "mean_residual",
            "num_observations",
        ]

        resid_df = pd.read_csv(csv_path, skiprows=2, names=resid_cols)

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

    def get_residual_gdfs(self):
        init_resid_path, final_resid_path = self.get_residual_csv_paths()
        init_resid_gdf = self.read_residuals_csv(init_resid_path)
        final_resid_gdf = self.read_residuals_csv(final_resid_path)
        return init_resid_gdf, final_resid_gdf


class PlotResiduals:
    def __init__(self, geodataframes):
        if not isinstance(geodataframes, list):
            raise ValueError("Input must be a list of GeoDataFrames")
        self.geodataframes = geodataframes

    def get_residual_stats(self, gdf, column_name="mean_residual"):
        stats = gdf[column_name].quantile([.25, .50, .84, .95]).round(2).tolist()
        return stats

    def plot_n_residuals(
        self,
        column_name="mean_residual",
        clip_final=True,
        lognorm=False,
        clim=None,
        cmap="inferno",
        map_crs="EPSG:4326",
        title_size=10,
        **ctx_kwargs,
    ):

        # Get rows and columns and create subplots
        n = len(self.geodataframes)
        nrows = (n + 3) // 4
        ncols = min(n, 4)
        if n == 1:
            f, axa = plt.subplots(1, 1, figsize=(8, 6))
            axa = [axa]
        else:
            f, axa = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True
            )
            axa = axa.flatten()
          
        # Plot each GeoDataFrame
        for i, gdf in enumerate(self.geodataframes):
            if clim is None:
                clim = ColorBar().get_clim(gdf[column_name])
            
            gdf = gdf.sort_values(by=column_name).to_crs(map_crs)
            plotter = Plotter(
                ax=axa[i],
                clim=clim,
                label=column_name,
            )

            plotter.plot_geodataframe(gdf, column_name, lognorm)

            ctx.add_basemap(ax=axa[i], **ctx_kwargs)

            if clip_final and i == n - 1:
                axa[i].autoscale(False)
            
            
            # Show some statistics and information
            axa[i].set_title(f"{column_name:} (n={gdf.shape[0]})", fontsize=title_size)

            stats = self.get_residual_stats(gdf, column_name)
            stats_text = "\n".join(f"{quantile*100:.0f}th perc: {stat}" for quantile, stat in zip([.25, .50, .84, .95], stats))
            axa[i].text(0.05, 0.95, stats_text, transform=axa[i].transAxes, fontsize=8,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Clean up axes and tighten layout
        for i in range(n, nrows * ncols):
            f.delaxes(axa[i])
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        plt.tight_layout()

