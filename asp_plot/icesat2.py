import logging
import os
import subprocess

import contextily as ctx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rioxarray
import xarray as xr
from sliderule import icesat2, sliderule

from asp_plot.utils import ColorBar, Plotter, save_figure

icesat2.init("slideruleearth.io", verbose=True)


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ICESat2(Plotter):
    def __init__(
        self,
        dem_fn,
        geojson_fn,
        aligned_dem_fn=None,
        atl06=None,
        atl06_clean=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dem_fn = dem_fn
        self.geojson_fn = geojson_fn
        if atl06 is not None and not isinstance(atl06, gpd.GeoDataFrame):
            raise ValueError("ATL06 must be a GeoDataFrame if provided.")
        self.atl06 = atl06
        if atl06_clean is not None and not isinstance(atl06_clean, gpd.GeoDataFrame):
            raise ValueError("Cleaned ATL06 must be a GeoDataFrame if provided.")
        self.atl06_clean = atl06_clean
        self.aligned_dem_fn = aligned_dem_fn

    def pull_atl06_data(
        self,
        esa_worldcover=True,
        save_to_gpkg=False,
        filename_to_save="atl06_all",
        srt=0,
        cnf=4,
        ats=5,
        cnt=5,
        len=40,
        res=20,
        maxi=5,
        H_min_win=3,
        sigma_r_max=5,
    ):
        if not os.path.exists(self.geojson_fn):
            raise ValueError(
                f"Geojson file not found: {self.geojson_fn}\nUse this tool to make and download one:\nhttps://geojson.io/"
            )

        region = sliderule.toregion(self.geojson_fn)["poly"]

        # Build ATL06 Request
        params = {
            "poly": region,
            "srt": srt,
            "cnf": cnf,
            "ats": ats,
            "cnt": cnt,
            "len": len,
            "res": res,
            "maxi": maxi,
            "H_min_win": H_min_win,
            "sigma_r_max": sigma_r_max,
        }

        if esa_worldcover:
            params["samples"] = {
                "esa-worldcover-10meter": {
                    "asset": "esa-worldcover-10meter",
                    "algorithm": "NearestNeighbour",
                },
            }

        # Make request
        print("\nICESat-2 ATL06 request processing\n")
        self.atl06 = icesat2.atl06p(params)

        if save_to_gpkg:
            self.atl06.to_file(f"{filename_to_save}.gpkg", driver="GPKG")

        return self.atl06

    def clean_atl06(
        self,
        h_sigma_quantile=0.95,
        mask_worldcover_water=True,
        select_months=None,
        select_years=None,
        save_to_csv=False,
        save_to_gpkg=False,
        filename_to_save="atl06_clean",
    ):
        def to_csv(atl06, fn_out):
            df = atl06[["geometry", "h_mean"]].copy()
            df["lon"] = df["geometry"].x
            df["lat"] = df["geometry"].y
            df["height_above_datum"] = df["h_mean"]
            df = df[["lon", "lat", "height_above_datum"]]
            df.to_csv(fn_out, header=True, index=False)

        # From Aimee Gibbons:
        # I'd recommend anything cycle 03 and later, due to pointing issues before cycle 03.
        self.atl06_clean = self.atl06[self.atl06["cycle"] >= 3]

        # Remove bad fits using high percentile of `h_sigma`, the error estimate for the least squares fit model.
        # Also need to filter out 0 values, not sure what these are caused by, but also very bad points.
        self.atl06_clean = self.atl06_clean[
            self.atl06_clean["h_sigma"]
            < self.atl06_clean["h_sigma"].quantile(h_sigma_quantile)
        ]
        self.atl06_clean = self.atl06_clean[self.atl06_clean["h_sigma"] != 0]

        # Clip to DEM area
        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        bounds = dem.rio.bounds()
        epsg = dem.rio.crs.to_epsg()
        bounds = rio.warp.transform_bounds(f"EPSG:{epsg}", "EPSG:4326", *bounds)
        self.atl06_clean = self.atl06_clean.cx[
            bounds[0] : bounds[2], bounds[1] : bounds[3]
        ]

        # Mask out water using ESA WorldCover (if it exists)
        # Value	Color	Description
        # 10	#006400	Tree cover
        # 20	#ffbb22	Shrubland
        # 30	#ffff4c	Grassland
        # 40	#f096ff	Cropland
        # 50	#fa0000	Built-up
        # 60	#b4b4b4	Bare / sparse vegetation
        # 70	#f0f0f0	Snow and ice
        # 80	#0064c8	Permanent water bodies
        # 90	#0096a0	Herbaceous wetland
        # 95	#00cf75	Mangroves
        # 100	#fae6a0	Moss and lichen
        if mask_worldcover_water:
            if "esa-worldcover-.value" not in self.atl06_clean.columns:
                logger.warning(
                    "\nESA WorldCover not found in ATL06 dataframe. Proceeding without water masking.\n"
                )
            else:
                self.atl06_clean = self.atl06_clean[
                    self.atl06_clean["esa-worldcover-.value"] != 80
                ]

        # Filter by time
        if select_months:
            self.atl06_clean = self.atl06_clean[
                self.atl06_clean.index.month.isin(select_months)
            ]
        if select_years:
            self.atl06_clean = self.atl06_clean[
                self.atl06_clean.index.year.isin(select_years)
            ]

        if save_to_csv:
            to_csv(self.atl06_clean, f"{filename_to_save}.csv")
        if save_to_gpkg:
            self.atl06_clean.to_file(f"{filename_to_save}.gpkg", driver="GPKG")

        return self.atl06_clean

    def plot_atl06(
        self,
        clean=False,
        plot_beams=False,
        use_dem_basemap=False,
        column_name="h_mean",
        cbar_label="Height above datum (m)",
        clim=None,
        cmap="inferno",
        map_crs="EPSG:4326",
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        if clean:
            atl06 = self.atl06_clean
        else:
            atl06 = self.atl06

        if clim is None:
            clim = ColorBar().get_clim(atl06[column_name])

        atl06_sorted = atl06.sort_values(by=column_name).to_crs(map_crs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        if use_dem_basemap:
            ctx_kwargs = {}
            # TODO: add rioxarray resampling and plotting to Plotter (plot_geo_array?)
            dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
            downscale_factor = 0.1
            new_width = dem.rio.width * downscale_factor
            new_height = dem.rio.height * downscale_factor
            dem_downsampled = dem.rio.reproject(
                dem.rio.crs,
                shape=(int(new_height), int(new_width)),
            )
            dem_downsampled.plot(ax=ax, cmap="inferno", add_colorbar=False, alpha=0.8)
            ax.set_title(None)

        if plot_beams:
            color_dict = {
                1: "red",
                2: "lightpink",
                3: "blue",
                4: "lightblue",
                5: "green",
                6: "lightgreen",
            }
            patches = [mpatches.Patch(color=v, label=k) for k, v in color_dict.items()]
            atl06_sorted.plot(
                ax=ax,
                markersize=1,
                color=atl06_sorted["spot"].map(color_dict).values,
            )
            ax.legend(
                handles=patches, title="laser spot\n(strong=1,3,5)", loc="upper left"
            )
            if ctx_kwargs:
                ctx.add_basemap(ax=ax, **ctx_kwargs)
        else:
            self.plot_geodataframe(
                ax=ax,
                gdf=atl06_sorted,
                column_name=column_name,
                cbar_label=cbar_label,
                cmap=cmap,
                clim=clim,
                **ctx_kwargs,
            )

        fig.suptitle(self.title, size=10)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def run_subprocess_command(self, command):
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line.strip())

        process.stdout.close()
        return_code = process.wait()
        if return_code == 0:
            print("\nCommand executed successfully.\n")
        else:
            print("\nCommand failed.\n")

    def pc_align_dem_to_atl06(
        self,
        max_displacement=20,
        max_source_points=10000000,
        alignment_method="point-to-point",
        atl06_csv=None,
        output_prefix=None,
    ):
        if atl06_csv is None or not os.path.exists(atl06_csv):
            raise ValueError(
                f"\nATL06 clean CSV file not found: {atl06_csv}\n\nWe need this to run pc_align.\n"
            )
        if not output_prefix:
            raise ValueError("\nPlease provide an output prefix for pc_align.\n")
        if self.aligned_dem_fn:
            logger.warning(
                f"\nAligned DEM already exists: {self.aligned_dem_fn}\n\nPlease use that, or remove this file before running pc_align.\n"
            )
            return

        command = [
            "pc_align",
            "--max-displacement",
            str(max_displacement),
            "--max-num-source-points",
            str(max_source_points),
            "--alignment-method",
            alignment_method,
            "--csv-format",
            "1:lon 2:lat 3:height_above_datum",
            "--compute-translation-only",
            "--save-inv-transformed-reference-points",
            "--output-prefix",
            output_prefix,
            self.dem_fn,
            atl06_csv,
        ]

        self.run_subprocess_command(command)

    def generate_translated_dem(self, pc_align_output, dem_out_fn):
        if not os.path.exists(pc_align_output):
            raise ValueError(
                f"\npc_align output not found: {pc_align_output}\n\nWe need this to generate the translated DEM.\n"
            )
        if self.aligned_dem_fn:
            logger.warning(
                f"\nAligned DEM already exists: {self.aligned_dem_fn}\n\nPlease use that, or remove this file before running pc_align.\n"
            )
            return

        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        epsg = dem.rio.crs.to_epsg()
        gsd = dem.rio.transform()[0]

        command = [
            "point2dem",
            "--tr",
            str(gsd),
            "--t_srs",
            f"EPSG:{epsg}",
            "--nodata-value",
            str(-9999),
            "-o",
            dem_out_fn,
            pc_align_output,
        ]

        self.run_subprocess_command(command)

    def compare_atl06_to_dem(
        self, use_aligned_dem=False, save_dir=None, fig_fn=None, **ctx_kwargs
    ):
        if self.atl06_clean is None:
            raise ValueError(
                "\nPlease clean ATL06 data with clean_atl06 function before comparing to DEM.\n"
            )

        if use_aligned_dem:
            print("\nUsing aligned DEM for comparison.\n")
            dem = rioxarray.open_rasterio(self.aligned_dem_fn, masked=True).squeeze()
        else:
            print(
                "\nComparing ATL06 to DEM. Gross mismatches or spatial trends may indicate a need for pc_align step.\n"
            )
            dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        epsg = dem.rio.crs.to_epsg()
        self.atl06_clean = self.atl06_clean.to_crs(f"EPSG:{epsg}")

        x = xr.DataArray(self.atl06_clean.geometry.x.values, dims="z")
        y = xr.DataArray(self.atl06_clean.geometry.y.values, dims="z")
        sample = dem.interp(x=x, y=y)

        self.atl06_clean["dem_height"] = sample.values
        self.atl06_clean["icesat_minus_dem"] = (
            self.atl06_clean["h_mean"] - self.atl06_clean["dem_height"]
        )

        clim = (
            float(self.atl06_clean["icesat_minus_dem"].quantile(0.05)),
            float(self.atl06_clean["icesat_minus_dem"].quantile(0.95)),
        )
        abs_max = max(abs(clim[0]), abs(clim[1]))
        clim = (-abs_max, abs_max)

        self.plot_atl06(
            clean=True,
            column_name="icesat_minus_dem",
            cbar_label="ICESat-2 minus DEM (m)",
            clim=clim,
            cmap="RdBu",
            map_crs=f"EPSG:{epsg}",
            save_dir=save_dir,
            fig_fn=fig_fn,
            **ctx_kwargs,
        )

    def atl06_dem_histogram(self, save_dir=None, fig_fn=None):
        if "icesat_minus_dem" not in self.atl06_clean.columns:
            raise ValueError(
                "\nNo icesat_minus_dem column found. Please run compare_atl06_to_dem first.\n"
            )

        def nmad(a, c=1.4826):
            return np.nanmedian(np.fabs(a - np.nanmedian(a))) * c

        med = self.atl06_clean["icesat_minus_dem"].quantile(0.50)
        nmad = self.atl06_clean[["icesat_minus_dem"]].apply(nmad).iloc[0]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        xmin = self.atl06_clean["icesat_minus_dem"].quantile(0.01)
        xmax = self.atl06_clean["icesat_minus_dem"].quantile(0.99)
        plot_kwargs = {"bins": 128, "alpha": 0.5, "range": (xmin, xmax)}
        self.atl06_clean.hist(
            ax=ax,
            column="icesat_minus_dem",
            label=f"Median={med:0.2f}, NMAD={nmad:0.2f}",
            **plot_kwargs,
        )
        ax.legend()
        ax.set_title(None)
        ax.set_xlabel("ICESat-2 - DEM (m)")

        fig.suptitle(self.title, size=10)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_atl06_dem_profiles(
        self,
        select_days=None,
        select_months=None,
        select_years=None,
        only_strong_beams=True,
        save_dir=None,
        fig_fn=None,
    ):

        atl06 = self.atl06_clean

        # Additional day, month, and year filtering
        if select_years:
            atl06 = atl06[atl06.index.year.isin(select_years)]
        if select_months:
            atl06 = atl06[atl06.index.month.isin(select_months)]
        if select_days:
            atl06 = atl06[atl06.index.day.isin(select_days)]

        # Get day of interest
        dates = atl06.index.strftime("%Y-%m-%d").unique()

        if dates.size > 1:
            logger.warning(
                f"\nYou are trying to plot {dates.size} ICESat-2 passes. Please apply additional day, month, and year filtering to get only one pass for plotting.\n"
            )
            return
        else:
            date = dates[0]

        atl06 = atl06[atl06.index.normalize() == date]

        # Get unique beam strength spot numbers
        spots = atl06.spot.unique()

        # Optionally, filter out weak beams (2, 4, 6)
        if only_strong_beams:
            spots = spots[spots % 2 == 1]

        # Plot the beams
        fig, axes = plt.subplots(spots.size, 1, figsize=(10, 12))
        axes = axes.flatten()
        for ii, spot in enumerate(spots):
            ax = axes[ii]
            spot_to_plot = atl06[atl06.spot == spot]
            along_track_dist = abs(spot_to_plot.x_atc - spot_to_plot.x_atc.max()) / 1000

            ax.scatter(
                along_track_dist,
                spot_to_plot.h_mean,
                color="black",
                s=5,
                marker="s",
                label="ICESat-2 ATL06",
            )
            ax.scatter(
                along_track_dist,
                spot_to_plot.dem_height,
                color="red",
                s=5,
                marker="o",
                label="DEM",
            )
            ax.set_axisbelow(True)
            ax.grid(0.3)
            ax.set_title(f"Laser Spot {spot:0.0f}")
            ax.set_xlabel("Distance along track (km)")
            ax.set_ylabel("Elevation (m HAE)")
            ax.legend()

        fig.suptitle(self.title)
        fig.subplots_adjust(hspace=0.3)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
