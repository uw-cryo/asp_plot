import logging
import os
import subprocess

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
    def __init__(self, dem_fn, geojson_fn, atl06=None, atl06_clean=None, **kwargs):
        super().__init__(**kwargs)
        self.dem_fn = dem_fn
        self.geojson_fn = geojson_fn
        if atl06 is not None and not isinstance(atl06, gpd.GeoDataFrame):
            raise ValueError("ATL06 must be a GeoDataFrame if provided.")
        self.atl06 = atl06
        if atl06_clean is not None and not isinstance(atl06_clean, gpd.GeoDataFrame):
            raise ValueError("Cleaned ATL06 must be a GeoDataFrame if provided.")
        self.atl06_clean = atl06_clean

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
        def save_to_csv(atl06, fn_out):
            df = atl06[["geometry", "h_mean"]].copy()
            df["lon"] = df["geometry"].x
            df["lat"] = df["geometry"].y
            df["height_above_datum"] = df["h_mean"]
            df = df[["lon", "lat", "height_above_datum"]]
            df.to_csv(fn_out, header=True, index=False)

        # TODO: optionally save to parquet and/or csv
        # parquet needs time in [ms] so some precision loss
        # atl06.index = atl06.index.astype("datetime64[ms]")
        # csv will only save lat/lon/height (and time if possible?)

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
            save_to_csv(self.atl06_clean, f"{filename_to_save}.csv")
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
            dem_downsampled.plot(ax=ax, cmap="viridis", add_colorbar=False, alpha=0.8)

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
            self.plot_geodataframe(
                ax=ax,
                gdf=atl06_sorted,
                column_name="spot",
                cmap=None,
                color=atl06_sorted["spot"].map(color_dict).values,
                **ctx_kwargs,
            )
            ax.legend(
                handles=patches, title="laser spot\n(strong=1,3,5)", loc="upper left"
            )
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

    def pc_align_dem_to_atl06(self, atl06_csv=None, output_prefix=None):
        if atl06_csv is None or not os.path.exists(atl06_csv):
            raise ValueError(
                f"\nATL06 clean CSV file not found: {atl06_csv}\n\nWe need this to run pc_align.\n"
            )
        if not output_prefix:
            raise ValueError("\nPlease provide an output prefix for pc_align.\n")

        command = [
            "pc_align",
            "--max-displacement",
            "20",
            "--max-num-source-points",
            "10000000",
            "--alignment-method",
            "point-to-point",
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

    def compare_atl06_to_dem(self, save_dir=None, fig_fn=None, **ctx_kwargs):
        if self.atl06_clean is None:
            raise ValueError(
                "\nPlease clean ATL06 data with clean_atl06 function before comparing to DEM.\n"
            )

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

    # TODO: profile plotting
