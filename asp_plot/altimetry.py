import logging
import os

import contextily as ctx
import geopandas as gpd
import geoutils as gu
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
import xarray as xr
from osgeo import gdal, osr
from sliderule import icesat2

from asp_plot.utils import (
    ColorBar,
    Raster,
    glob_file,
    run_subprocess_command,
    save_figure,
)

icesat2.init("slideruleearth.io", verbose=True)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Altimetry:
    def __init__(
        self,
        dem_fn,
        aligned_dem_fn=None,
        atl06sr=None,
        atl06sr_filtered=None,
        **kwargs,
    ):
        if not os.path.exists(dem_fn):
            raise ValueError(f"DEM file not found: {dem_fn}")
        self.dem_fn = dem_fn
        if atl06sr is not None and not isinstance(atl06sr, gpd.GeoDataFrame):
            raise ValueError("ATL06 must be a GeoDataFrame if provided.")
        self.atl06sr = atl06sr
        if atl06sr_filtered is not None and not isinstance(
            atl06sr_filtered, gpd.GeoDataFrame
        ):
            raise ValueError("Cleaned ATL06 must be a GeoDataFrame if provided.")
        self.atl06sr_filtered = atl06sr_filtered
        self.aligned_dem_fn = aligned_dem_fn

    def pull_atl06sr(
        self,
        esa_worldcover=True,
        save_to_parquet=True,
        filename="atl06sr_defaults_all",
        parms=None,
    ):

        region = Raster(self.dem_fn).get_bounds(latlon=True)

        if parms is None:
            parms = {
                "poly": region,
                "cnf": 4,
            }
        else:
            parms["poly"] = region

        if esa_worldcover:
            parms["samples"] = {
                "esa_worldcover": {
                    "asset": "esa-worldcover-10meter",
                }
            }

        print(f"\nICESat-2 ATL06 request processing with parms:\n{parms}")
        self.atl06sr = icesat2.atl06p(parms)

        if save_to_parquet:
            # Need to write out this way instead of including option
            # in parms due to: https://github.com/SlideRuleEarth/sliderule/issues/298
            self.atl06sr.to_parquet(f"{filename}.parquet")

        return self.atl06sr

    def filter_atl06sr(
        self,
        h_sigma_quantile=0.95,
        mask_worldcover_water=True,
        select_years=None,
        select_months=None,
        select_days=None,
        save_to_csv=True,
        save_to_parquet=True,
        filename="atl06sr_defaults_filtered",
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
        self.atl06sr_filtered = self.atl06sr[self.atl06sr["cycle"] >= 3]

        # Remove bad fits using high percentile of `h_sigma`, the error estimate for the least squares fit model.
        # Also need to filter out 0 values, not sure what these are caused by, but also very bad points.
        self.atl06sr_filtered = self.atl06sr_filtered[
            self.atl06sr_filtered["h_sigma"]
            < self.atl06sr_filtered["h_sigma"].quantile(h_sigma_quantile)
        ]
        self.atl06sr_filtered = self.atl06sr_filtered[
            self.atl06sr_filtered["h_sigma"] != 0
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
            if "esa_worldcover.value" not in self.atl06sr_filtered.columns:
                logger.warning(
                    "\nESA WorldCover not found in ATL06 dataframe. Proceeding without water masking.\n"
                )
            else:
                self.atl06sr_filtered = self.atl06sr_filtered[
                    self.atl06sr_filtered["esa_worldcover.value"] != 80
                ]

        # Filter by time
        if select_years:
            self.atl06sr_filtered = self.atl06sr_filtered[
                self.atl06sr_filtered.index.year.isin(select_years)
            ]
        if select_months:
            self.atl06sr_filtered = self.atl06sr_filtered[
                self.atl06sr_filtered.index.month.isin(select_months)
            ]
        if select_days:
            self.atl06sr_filtered = self.atl06sr_filtered[
                self.atl06sr_filtered.index.day.isin(select_days)
            ]

        if save_to_csv:
            # Used for pc_align step
            to_csv(self.atl06sr_filtered, f"{filename}.csv")
        if save_to_parquet:
            # Need to write out this way instead of including option
            # in parms due to: https://github.com/SlideRuleEarth/sliderule/issues/298
            self.atl06sr_filtered.to_parquet(f"{filename}.parquet")

        return self.atl06sr_filtered

    def plot_atl06sr(
        self,
        filtered=False,
        plot_beams=False,
        plot_dem=False,
        column_name="h_mean",
        cbar_label="Height above datum (m)",
        title="ICESat-2 ATL06-SR",
        clim=None,
        symm_clim=False,
        cmap="inferno",
        map_crs="4326",
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        if filtered:
            atl06sr = self.atl06sr_filtered
        else:
            atl06sr = self.atl06sr

        atl06sr_sorted = atl06sr.sort_values(by=column_name).to_crs(f"EPSG:{map_crs}")

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        if plot_dem:
            ctx_kwargs = {}
            dem_downsampled = gu.Raster(self.dem_fn, downsample=10)
            cb = ColorBar(perc_range=(2, 98))
            cb.get_clim(dem_downsampled.data)
            dem_downsampled.plot(
                ax=ax,
                cmap="inferno",
                add_cbar=False,
                vmin=cb.clim[0],
                vmax=cb.clim[1],
                alpha=1,
            )
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
            atl06sr_sorted.plot(
                ax=ax,
                markersize=1,
                color=atl06sr_sorted["spot"].map(color_dict).values,
            )
            ax.legend(
                handles=patches, title="laser spot\n(strong=1,3,5)", loc="upper left"
            )
        else:
            if plot_dem:
                cb.symm = symm_clim
            else:
                cb = ColorBar(perc_range=(2, 98), symm=symm_clim)
                if clim is None:
                    cb.get_clim(atl06sr_sorted[column_name])
                else:
                    cb.clim = clim

            norm = cb.get_norm(lognorm=False)

            atl06sr_sorted.plot(
                ax=ax,
                column=column_name,
                cmap=cmap,
                norm=norm,
                s=1,
                legend=True,
                legend_kwds={"label": cbar_label},
            )

        if ctx_kwargs:
            ctx.add_basemap(ax=ax, **ctx_kwargs)

        fig.suptitle(title, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def pc_align_dem_to_atl06sr(
        self,
        max_displacement=20,
        max_source_points=10000000,
        alignment_method="point-to-point",
        atl06sr_csv=None,
        output_prefix=None,
    ):
        if atl06sr_csv is None or not os.path.exists(atl06sr_csv):
            raise ValueError(
                f"\nATL06 filtered CSV file not found: {atl06sr_csv}\n\nWe need this to run pc_align.\n"
            )
        if not output_prefix:
            raise ValueError("\nPlease provide an output prefix for pc_align.\n")
        if self.aligned_dem_fn:
            logger.warning(
                f"\nAligned DEM was already supplied: {self.aligned_dem_fn}\n\nPlease use that, or remove this file before running pc_align.\n"
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
            "--output-prefix",
            output_prefix,
            self.dem_fn,
            atl06sr_csv,
        ]

        run_subprocess_command(command)

    def pc_align_report(self, pc_align_folder="pc_align"):
        pc_align_log = glob_file(pc_align_folder, "*log-pc_align*.txt")

        with open(pc_align_log, "r") as file:
            content = file.readlines()

        report = ""
        for line in content:
            if "Input: error percentile of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Input: mean of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Output: error percentile of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Output: mean of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Translation vector (Cartesian, meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Translation vector magnitude (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line

        return report

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

        rast = Raster(self.dem_fn)
        gsd = rast.get_gsd()
        epsg = rast.get_epsg_code()

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

        run_subprocess_command(command)

    def apply_dem_translation(self, pc_align_folder="pc_align", inv_trans=True):
        def get_proj_shift(src_c, src_shift, s_srs, t_srs, inv_trans=True):
            if s_srs.IsSame(t_srs):
                proj_shift = src_shift
            else:
                src_c_shift = src_c + src_shift
                src2proj = osr.CoordinateTransformation(s_srs, t_srs)
                proj_c = np.array(src2proj.TransformPoint(*src_c))
                proj_c_shift = np.array(src2proj.TransformPoint(*src_c_shift))
                if inv_trans:
                    proj_shift = proj_c - proj_c_shift
                else:
                    proj_shift = proj_c_shift - proj_c
            # Reduce unnecessary precision
            proj_shift = np.around(proj_shift, 3)
            return proj_shift

        if self.aligned_dem_fn:
            logger.warning(
                f"\nAligned DEM already exists: {self.aligned_dem_fn}\n\nPlease use that, or remove this file before running pc_align.\n"
            )
            return

        pc_align_log = glob_file(pc_align_folder, "*log-pc_align*.txt")

        src = Raster(self.dem_fn)
        src_a = src.read_array()
        src_ndv = src.get_ndv()

        # Need to extract from log to know how to compute translation
        # if ref is csv and src is dem, want to transform source_center + shift
        # if ref is dem and src is csv, want to inverse transform ref by shift applied at (source_center - shift)

        llz_c = None
        with open(pc_align_log, "r") as file:
            content = file.readlines()

        for line in content:
            if "Centroid of source points (Cartesian, meters):" in line:
                ecef_c = np.genfromtxt([line.split("Vector3")[1][1:-2]], delimiter=",")
            if "Centroid of source points (lat,lon,z):" in line:
                llz_c = np.genfromtxt([line.split("Vector3")[1][1:-2]], delimiter=",")
            if "Translation vector (Cartesian, meters):" in line:
                ecef_shift = np.genfromtxt(
                    [line.split("Vector3")[1][1:-2]], delimiter=","
                )
            if "Translation vector (lat,lon,z):" in line:
                llz_shift = np.genfromtxt(
                    [line.split("Vector3")[1][1:-2]], delimiter=","
                )
                break

        if llz_c is None:
            raise ValueError(
                f"\nLog file does not contain necessary translation information: {pc_align_log}\n"
            )

        # Reorder lat,lon,z to lon,lat,z (x,y,z)
        i = [1, 0, 2]
        llz_c = llz_c[i]
        llz_shift = llz_shift[i]

        ecef_srs = osr.SpatialReference()
        ecef_srs.ImportFromEPSG(4978)

        s_srs = ecef_srs
        src_c = ecef_c
        src_shift = ecef_shift

        # Determine shift in original dataset coords
        t_srs = osr.SpatialReference()
        t_srs.ImportFromWkt(src.ds.crs.to_wkt())
        proj_shift = get_proj_shift(src_c, src_shift, s_srs, t_srs, inv_trans)

        self.aligned_dem_fn = self.dem_fn.replace(".tif", "_pc_align_translated.tif")
        gdal_opt = ["COMPRESS=LZW", "TILED=YES", "PREDICTOR=3", "BIGTIFF=IF_SAFER"]
        dst_ds = gdal.GetDriverByName("GTiff").CreateCopy(
            self.aligned_dem_fn, gdal.Open(self.dem_fn), strict=0, options=gdal_opt
        )
        # Apply vertical shift
        dst_b = dst_ds.GetRasterBand(1)
        dst_b.SetNoDataValue(float(src_ndv))
        dst_b.WriteArray(np.around((src_a + proj_shift[2]).filled(src_ndv), decimals=3))

        dst_gt = list(dst_ds.GetGeoTransform())
        # Apply horizontal shift directly to geotransform
        dst_gt[0] += proj_shift[0]
        dst_gt[3] += proj_shift[1]
        dst_ds.SetGeoTransform(dst_gt)

        print(f"\nWriting out: {self.aligned_dem_fn}\n")
        dst_ds = None

    def atl06sr_to_dem_dh(self):
        if self.atl06sr_filtered is None:
            raise ValueError(
                "\nPlease filter ATL06 data with filter_atl06sr function before comparing to DEM.\n"
            )

        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        epsg = dem.rio.crs.to_epsg()
        self.atl06sr_filtered = self.atl06sr_filtered.to_crs(f"EPSG:{epsg}")

        x = xr.DataArray(self.atl06sr_filtered.geometry.x.values, dims="z")
        y = xr.DataArray(self.atl06sr_filtered.geometry.y.values, dims="z")
        sample = dem.interp(x=x, y=y)

        self.atl06sr_filtered["dem_height"] = sample.values
        self.atl06sr_filtered["icesat_minus_dem"] = (
            self.atl06sr_filtered["h_mean"] - self.atl06sr_filtered["dem_height"]
        )

        if self.aligned_dem_fn:
            dem = rioxarray.open_rasterio(self.aligned_dem_fn, masked=True).squeeze()
            x = xr.DataArray(self.atl06sr_filtered.geometry.x.values, dims="z")
            y = xr.DataArray(self.atl06sr_filtered.geometry.y.values, dims="z")
            sample = dem.interp(x=x, y=y)

            self.atl06sr_filtered["dem_aligned_height"] = sample.values
            self.atl06sr_filtered["icesat_minus_dem_aligned"] = (
                self.atl06sr_filtered["h_mean"]
                - self.atl06sr_filtered["dem_aligned_height"]
            )

    def mapview_plot_atl06sr_to_dem(
        self, clim=None, plot_aligned=False, save_dir=None, fig_fn=None, **ctx_kwargs
    ):
        if plot_aligned:
            column_name = "icesat_minus_dem_aligned"
        else:
            column_name = "icesat_minus_dem"

        if column_name not in self.atl06sr_filtered.columns:
            self.atl06sr_to_dem_dh()

        if clim is not None:
            symm_clim = False
        else:
            symm_clim = True

        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        epsg = dem.rio.crs.to_epsg()

        self.plot_atl06sr(
            filtered=True,
            column_name=column_name,
            cbar_label="ICESat-2 minus DEM (m)",
            clim=clim,
            symm_clim=symm_clim,
            cmap="RdBu",
            map_crs=epsg,
            save_dir=save_dir,
            fig_fn=fig_fn,
            **ctx_kwargs,
        )

    def histogram(
        self, title="Histogram", plot_aligned=False, save_dir=None, fig_fn=None
    ):
        if "icesat_minus_dem" not in self.atl06sr_filtered.columns:
            self.atl06sr_to_dem_dh()

        def _nmad(a, c=1.4826):
            return np.nanmedian(np.fabs(a - np.nanmedian(a))) * c

        column_names = ["icesat_minus_dem"]
        if plot_aligned:
            column_names.append("icesat_minus_dem_aligned")

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for column_name in column_names:
            med = self.atl06sr_filtered[column_name].quantile(0.50)
            nmad = self.atl06sr_filtered[[column_name]].apply(_nmad).iloc[0]

            xmin = self.atl06sr_filtered[column_name].quantile(0.01)
            xmax = self.atl06sr_filtered[column_name].quantile(0.99)
            plot_kwargs = {"bins": 128, "alpha": 0.5, "range": (xmin, xmax)}
            self.atl06sr_filtered.hist(
                ax=ax,
                column=column_name,
                label=f"{column_name}, Median={med:0.2f}, NMAD={nmad:0.2f}",
                **plot_kwargs,
            )

        ax.legend()
        ax.set_title(None)
        ax.set_xlabel("ICESat-2 - DEM (m)")

        fig.suptitle(title, size=10)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_atl06sr_dem_profiles(
        self,
        title="ICESat-2 ATL06-SR Profiles",
        select_years=None,
        select_months=None,
        select_days=None,
        only_strong_beams=True,
        save_dir=None,
        fig_fn=None,
    ):
        if "icesat_minus_dem" not in self.atl06sr_filtered.columns:
            self.atl06sr_to_dem_dh()

        atl06sr = self.atl06sr_filtered

        # Additional day, month, and year filtering
        if select_years:
            atl06sr = atl06sr[atl06sr.index.year.isin(select_years)]
        if select_months:
            atl06sr = atl06sr[atl06sr.index.month.isin(select_months)]
        if select_days:
            atl06sr = atl06sr[atl06sr.index.day.isin(select_days)]

        # Get day of interest
        dates = atl06sr.index.strftime("%Y-%m-%d").unique()

        if dates.size > 1:
            logger.warning(
                f"\nYou are trying to plot {dates.size} ICESat-2 passes. Please apply additional day, month, and year filtering to get only one pass for plotting.\n"
            )
            return
        else:
            date = dates[0]

        atl06sr = atl06sr[atl06sr.index.normalize() == date]

        # Get unique beam strength spot numbers
        spots = atl06sr.spot.unique()

        # Optionally, filter out weak beams (2, 4, 6)
        if only_strong_beams:
            spots = spots[spots % 2 == 1]

        # Plot the beams
        fig, axes = plt.subplots(spots.size, 1, figsize=(10, 12))
        axes = axes.flatten()
        for ii, spot in enumerate(spots):
            ax = axes[ii]
            spot_to_plot = atl06sr[atl06sr.spot == spot]
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
                spot_to_plot.dem_aligned_height,
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

        fig.suptitle(title)
        fig.subplots_adjust(hspace=0.3)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
