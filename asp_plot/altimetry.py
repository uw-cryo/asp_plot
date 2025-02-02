import logging
import os

import contextily as ctx
import geopandas as gpd
import geoutils as gu
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from sliderule import icesat2

from asp_plot.alignment import Alignment
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import ColorBar, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class Altimetry:
    def __init__(
        self,
        directory,
        dem_fn,
        aligned_dem_fn=None,
        atl06sr_processing_levels={},
        atl06sr_processing_levels_filtered={},
        # atl03sr=None,
        **kwargs,
    ):
        self.directory = directory

        if not os.path.exists(dem_fn):
            raise ValueError(f"DEM file not found: {dem_fn}")
        self.dem_fn = dem_fn

        if aligned_dem_fn is not None and not os.path.exists(aligned_dem_fn):
            raise ValueError(f"Aligned DEM file not found: {aligned_dem_fn}")
        self.aligned_dem_fn = aligned_dem_fn

        self.atl06sr_processing_levels = atl06sr_processing_levels
        self.atl06sr_processing_levels_filtered = atl06sr_processing_levels_filtered

        #Initialize the SlideRule session (requires active connection)
        icesat2.init("slideruleearth.io", verbose=True)

        # TODO: Implement alongside request_atl03sr below
        # if atl03sr is not None and not isinstance(atl03sr, gpd.GeoDataFrame):
        #     raise ValueError("ATL03 must be a GeoDataFrame if provided.")
        # self.atl03sr = atl03sr

    # TODO: Implement ATL03 pull, which needs to put in separate GDF; warning this is gonna be huge and only used for basic plots
    # def request_atl03sr(self, rgt, cycle, track, spot, save_to_parquet=False, filename="atl03sr_defaults"):
    #     region = Raster(self.dem_fn).get_bounds(latlon=True)

    #     parms = {
    #         "poly": region,
    #         # classification and checks
    #         "pass_invalid": True, # still return photon segments that fail checks
    #         "cnf": -2, # all photons
    #         "atl08_class": ["atl08_noise", "atl08_ground", "atl08_canopy", "atl08_top_of_canopy", "atl08_unclassified"],
    #         #"yapc": {"score": 0}, # all photons
    #         # track selection
    #         "rgt": rgt,
    #         "cycle": cycle,
    #         "track": track,
    #         "spot": spot,
    #     }

    #     print(f"\nICESat-2 ATL03 request processing with parms:\n{parms}")
    #     self.atl03sr = icesat2.atl03sp(parms)

    #     if save_to_parquet:
    #         # Need to write out this way instead of including option
    #         # in parms due to:
    #         self.atl03sr.to_parquet(f"{filename}.parquet")

    #     return self.atl03sr

    def request_atl06sr_multi_processing(
        self,
        processing_levels=["all", "ground", "canopy", "top_of_canopy"],
        res=20,
        len=40,
        ats=20,
        cnt=10,
        maxi=6,
        h_sigma_quantile=1.0,
        save_to_parquet=False,
        filename="atl06sr",
        region=None,
    ):
        if not region:
            region = Raster(self.dem_fn).get_bounds(latlon=True)

        # See parameter discussion on: https://github.com/SlideRuleEarth/sliderule/issues/448
        # "srt": -1 tells the server side code to look at the ATL03 confidence array for each photon
        # and choose the confidence level that is highest across all five surface type entries.
        # cnf options: {"atl03_tep", "atl03_not_considered", "atl03_background", "atl03_within_10m", \
        # "atl03_low", "atl03_medium", "atl03_high"}
        # Note reduce count for limited number of ground photons

        # TODO: isolate shared and custom parms, initialize each with shared parameters, then set custom
        # TODO: use the WorldCover values to determine if we should report canopy or top of canopy
        # TODO: Use more generic variable names and strings for functions that are not just limited to atl06
        parms_dict = {
            "all": {
                "cnf": "atl03_high",
                "srt": -1,
                "cnt": cnt,
            },
            "ground": {
                "cnf": "atl03_low",
                "srt": -1,
                "cnt": 5,
                "atl08_class": "atl08_ground",
            },
            "canopy": {
                "cnf": "atl03_medium",
                "srt": -1,
                "cnt": 5,
                "atl08_class": ["atl08_canopy", "atl08_top_of_canopy"],
            },
            "top_of_canopy": {
                "cnf": "atl03_medium",
                "srt": -1,
                "cnt": 5,
                "atl08_class": "atl08_top_of_canopy",
            },
        }

        parms_dict = {
            key: parms for key, parms in parms_dict.items() if key in processing_levels
        }

        for key, parms in parms_dict.items():
            parms["poly"] = region
            parms["res"] = res
            parms["len"] = len
            parms["ats"] = ats
            parms["maxi"] = maxi
            parms["samples"] = {
                "esa_worldcover": {
                    "asset": "esa-worldcover-10meter",
                }
            }

            # TODO: write out relevant parameters in metadata, not filename
            #fn_base = f"{filename}_res{res}_len{len}_cnt{cnt}_ats{ats}_maxi{maxi}_{key}"
            fn_base = f"{filename}_{key}"

            print(f"\nICESat-2 ATL06 request processing for: {key}")
            fn = f"{fn_base}.parquet"

            print(parms)

            if os.path.exists(fn):
                print(f"Existing file found, reading in: {fn}")
                atl06sr = gpd.read_parquet(fn)
                #TODO: check that parms in file are same as request 
            else:
                #This is the actual SlideRule call!
                atl06sr = icesat2.atl06p(parms)
                if save_to_parquet:
                    atl06sr.to_parquet(fn)

            self.atl06sr_processing_levels[key] = atl06sr

            print(f"Filtering ATL06-SR {key}")
            fn = f"{fn_base}_filtered.parquet"

            if os.path.exists(fn):
                print(f"Existing file found, reading in: {fn}")
                atl06sr_filtered = gpd.read_parquet(fn)
            else:
                # From Aimee Gibbons:
                # I'd recommend anything cycle 03 and later, due to pointing issues before cycle 03.
                atl06sr_filtered = atl06sr[atl06sr["cycle"] >= 3]

                # Remove bad fits using high percentile of `h_sigma`, the error estimate for the least squares fit model.
                # TODO: not sure about h_sigma quantile...might throw out too much. Maybe just remove 0 values?
                atl06sr_filtered = atl06sr_filtered[
                    atl06sr_filtered["h_sigma"]
                    < atl06sr_filtered["h_sigma"].quantile(h_sigma_quantile)
                ]
                # Also need to filter out 0 values, not sure what these are caused by, but also very bad points.
                atl06sr_filtered = atl06sr_filtered[atl06sr_filtered["h_sigma"] != 0]

                if save_to_parquet:
                    atl06sr_filtered.to_parquet(fn)

            self.atl06sr_processing_levels_filtered[key] = atl06sr_filtered

    def filter_esa_worldcover(self, filter_out="water", retain_only=None):
        # Value	Description
        # 10	  Tree cover
        # 20	  Shrubland
        # 30	  Grassland
        # 40	  Cropland
        # 50	  Built-up
        # 60	  Bare / sparse vegetation
        # 70	  Snow and ice
        # 80	  Permanent water bodies
        # 90	  Herbaceous wetland
        # 95	  Mangroves
        # 100	  Moss and lichen
        value_dict = {
            "water": [80],
            "snow_ice": [70],
            "trees": [10],
            "low_vegetation": [20, 30, 40, 90, 95, 100],
            "built_up": [50],
        }

        if retain_only is not None:
            if retain_only in value_dict:
                values_to_keep = value_dict[retain_only]
                for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
                    if "esa_worldcover.value" in atl06sr.columns:
                        mask = atl06sr["esa_worldcover.value"].isin(values_to_keep)
                        self.atl06sr_processing_levels_filtered[key] = atl06sr[mask]
            else:
                logger.warning(
                    f"\nESA WorldCover retain value not found: {retain_only}\n"
                )
                return

        elif filter_out in value_dict:
            values_to_filter = value_dict[filter_out]
            for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
                if "esa_worldcover.value" in atl06sr.columns:
                    mask = ~atl06sr["esa_worldcover.value"].isin(values_to_filter)
                    self.atl06sr_processing_levels_filtered[key] = atl06sr[mask]
        else:
            logger.warning(f"\nESA WorldCover filter value not found: {filter_out}\n")
            return

    def predefined_temporal_filter_atl06sr(self, date=None):
        if date is None:
            date = StereopairMetadataParser(self.directory).get_pair_dict()["cdate"]

        original_keys = list(self.atl06sr_processing_levels_filtered.keys())

        for key in original_keys:
            print(
                f"\nFiltering ATL06 with 15 day pad, 90 day pad, and seasonal pad around {date} for: {key}"
            )
            atl06sr = self.atl06sr_processing_levels_filtered[key]

            fifteen_day = atl06sr[abs(atl06sr.index - date) <= pd.Timedelta(days=15)]
            fortyfive_day = atl06sr[abs(atl06sr.index - date) <= pd.Timedelta(days=45)]

            image_season = date.strftime("%b")
            if image_season in ["Dec", "Jan", "Feb"]:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Dec", "Jan", "Feb"])
                ]
            elif image_season in ["Mar", "Apr", "May"]:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Mar", "Apr", "May"])
                ]
            elif image_season in ["Jun", "Jul", "Aug"]:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Jun", "Jul", "Aug"])
                ]
            else:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Sep", "Oct", "Nov"])
                ]

            if not fifteen_day.empty:
                self.atl06sr_processing_levels_filtered[f"{key}_15_day_pad"] = (
                    fifteen_day
                )
            if not fortyfive_day.empty:
                self.atl06sr_processing_levels_filtered[f"{key}_45_day_pad"] = (
                    fortyfive_day
                )
            if not season_filter.empty:
                self.atl06sr_processing_levels_filtered[f"{key}_seasonal"] = (
                    season_filter
                )

    def generic_temporal_filter_atl06sr(
        self, select_years=None, select_months=None, select_days=None
    ):
        for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
            print(f"\nFiltering ATL06 for: {key}")
            atl06_filtered = atl06sr

            if select_years:
                atl06_filtered = atl06_filtered[
                    atl06_filtered.index.year.isin(select_years)
                ]
            if select_months:
                atl06_filtered = atl06_filtered[
                    atl06_filtered.index.month.isin(select_months)
                ]
            if select_days:
                atl06_filtered = atl06_filtered[
                    atl06_filtered.index.day.isin(select_days)
                ]

            self.atl06sr_processing_levels_filtered[key] = atl06_filtered

    def to_csv_for_pc_align(self, key="ground", filename_prefix="atl06sr_for_pc_align"):
        atl06sr = self.atl06sr_processing_levels_filtered[key].to_crs("EPSG:4326")
        csv_fn = f"{filename_prefix}_{key}.csv"
        df = atl06sr[["geometry", "h_mean"]].copy()
        df["lon"] = df["geometry"].x
        df["lat"] = df["geometry"].y
        df["height_above_datum"] = df["h_mean"]
        df = df[["lon", "lat", "height_above_datum"]]
        df.to_csv(csv_fn, header=True, index=False)
        return csv_fn

    def alignment_report(
        self,
        processing_level="ground",
        minimum_points=500,
        agreement_threshold=0.25,
        write_out_aligned_dem=False,
        min_translation_threshold=0.1,
        key_for_aligned_dem="ground",
    ):
        filtered_keys = [
            key
            for key in self.atl06sr_processing_levels_filtered.keys()
            if key.startswith(processing_level)
        ]

        alignment = Alignment(self.directory, self.dem_fn)

        for key in filtered_keys:
            atl06sr = self.atl06sr_processing_levels_filtered[key]
            if atl06sr.shape[0] < minimum_points:
                print(
                    f"\n{key} has {atl06sr.shape[0]} points, which is less than the suggested {minimum_points} points. Skipping alignment.\n"
                )
                continue

            if not glob_file(
                os.path.join(self.directory, "pc_align"), f"*{key}*transform.txt"
            ):
                csv_fn = self.to_csv_for_pc_align(key=key)

                print(f"\nAligning {key} to DEM with pc_align\n")

                alignment.pc_align_dem_to_atl06sr(
                    atl06sr_csv=csv_fn,
                    output_prefix=f"pc_align/pc_align_{key}",
                )

        report_data = []
        for key in filtered_keys:
            report = alignment.pc_align_report(output_prefix=f"pc_align/pc_align_{key}")
            report_data.append({"key": key} | report)
        alignment_report_df = pd.DataFrame(report_data)

        gsd = Raster(self.dem_fn).get_gsd()
        if (
            alignment_report_df["translation_magnitude"].mean()
            < min_translation_threshold * gsd
        ):
            write_out_aligned_dem = False
            print(
                f"\nTranslation magnitude is less than {min_translation_threshold*100}% of the DEM GSD. Skipping writing out aligned DEM.\n"
            )

        if write_out_aligned_dem:
            # Calculate ranges and mean for each shift component
            x_range = (
                alignment_report_df["x_shift"].max()
                - alignment_report_df["x_shift"].min()
            )
            y_range = (
                alignment_report_df["y_shift"].max()
                - alignment_report_df["y_shift"].min()
            )
            z_range = (
                alignment_report_df["z_shift"].max()
                - alignment_report_df["z_shift"].min()
            )
            x_mean = alignment_report_df["x_shift"].mean()
            y_mean = alignment_report_df["y_shift"].mean()
            z_mean = alignment_report_df["z_shift"].mean()

            # Check if range is more than X% of mean for any component
            if (
                x_range > abs(x_mean * agreement_threshold)
                or y_range > abs(y_mean * agreement_threshold)
                or z_range > abs(z_mean * agreement_threshold)
            ):
                print(
                    f"\nWarning: Translation components vary by more than {agreement_threshold*100}% across temporal filters. The translation applied to the aligned DEM may be inaccurate.\n"
                )
                print(f"X shift range: {x_range:.3f} m (mean: {x_mean:.3f} m)")
                print(f"Y shift range: {y_range:.3f} m (mean: {y_mean:.3f} m)")
                print(f"Z shift range: {z_range:.3f} m (mean: {z_mean:.3f} m)")

            self.aligned_dem_fn = alignment.apply_dem_translation(
                output_prefix=f"pc_align/pc_align_{key_for_aligned_dem}",
            )
            print(
                f"\nWrote out {key_for_aligned_dem} aligned DEM to {self.aligned_dem_fn}\n"
            )

        self.alignment_report_df = alignment_report_df

    def plot_atl06sr_time_stamps(
        self,
        key="all",
        title="ICESat-2 ATL06-SR Time Stamps",
        cmap="inferno",
        map_crs="4326",
        figsize=(15, 10),
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        time_stamps = ["", "_15_day_pad", "_45_day_pad", "_seasonal"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        x_bounds = []
        y_bounds = []
        for ax, time_stamp in zip(axes, time_stamps):
            key_to_plot = f"{key}{time_stamp}"

            if key_to_plot not in self.atl06sr_processing_levels_filtered.keys():
                ax.text(
                    0.5,
                    0.5,
                    f"No points found for {key_to_plot}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            atl06sr = self.atl06sr_processing_levels_filtered[key_to_plot]
            if map_crs:
                crs = f"EPSG:{map_crs}"
                ctx_kwargs["crs"] = f"EPSG:{map_crs}"
            elif ctx_kwargs:
                crs = ctx_kwargs["crs"]
            else:
                crs = "EPSG:4326"
            atl06sr_sorted = atl06sr.sort_values(by="h_mean").to_crs(crs)
            bounds = atl06sr_sorted.total_bounds
            x_bounds.extend([bounds[0], bounds[2]])
            y_bounds.extend([bounds[1], bounds[3]])

            cb = ColorBar(perc_range=(2, 98))
            cb.get_clim(atl06sr_sorted["h_mean"])
            norm = cb.get_norm(lognorm=False)

            atl06sr_sorted.plot(
                ax=ax,
                column="h_mean",
                cmap=cmap,
                norm=norm,
                s=1,
                legend=True,
                legend_kwds={"label": "Height above datum (m)"},
            )

            ax.set_title(f"{key_to_plot} (n={atl06sr.shape[0]})", size=12)

        # 5% padding
        padding = 0.05
        x_range = max(x_bounds) - min(x_bounds)
        y_range = max(y_bounds) - min(y_bounds)
        for ax, time_stamp in zip(axes, time_stamps):
            key_to_plot = f"{key}{time_stamp}"

            if key_to_plot not in self.atl06sr_processing_levels_filtered.keys():
                continue

            ax.set_xlim(
                min(x_bounds) - padding * x_range, max(x_bounds) + padding * x_range
            )
            ax.set_ylim(
                min(y_bounds) - padding * y_range, max(y_bounds) + padding * y_range
            )
            if ctx_kwargs:
                ctx.add_basemap(ax=ax, **ctx_kwargs)

        fig.suptitle(f"{title}", size=14)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_atl06sr(
        self,
        key="all",
        plot_beams=False,
        plot_dem=False,
        column_name="h_mean",
        cbar_label="Height above datum (m)",
        title="ICESat-2 ATL06-SR",
        clim=None,
        symm_clim=False,
        cmap="inferno",
        map_crs="4326",
        figsize=(6, 4),
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        atl06sr = self.atl06sr_processing_levels_filtered[key]
        atl06sr_sorted = atl06sr.sort_values(by=column_name).to_crs(f"EPSG:{map_crs}")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

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

        ax.set_xticks([])
        ax.set_yticks([])

        fig.suptitle(f"{title}\n{key} (n={atl06sr.shape[0]})", size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def atl06sr_to_dem_dh(self):
        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        epsg = dem.rio.crs.to_epsg()
        for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
            atl06sr = atl06sr.to_crs(f"EPSG:{epsg}")

            x = xr.DataArray(atl06sr.geometry.x.values, dims="z")
            y = xr.DataArray(atl06sr.geometry.y.values, dims="z")
            sample = dem.interp(x=x, y=y)

            atl06sr["dem_height"] = sample.values
            atl06sr["icesat_minus_dem"] = atl06sr["h_mean"] - atl06sr["dem_height"]
            self.atl06sr_processing_levels_filtered[key] = atl06sr

        if self.aligned_dem_fn:
            aligned_dem = rioxarray.open_rasterio(
                self.aligned_dem_fn, masked=True
            ).squeeze()
            epsg = aligned_dem.rio.crs.to_epsg()
            for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
                atl06sr = atl06sr.to_crs(f"EPSG:{epsg}")

                x = xr.DataArray(atl06sr.geometry.x.values, dims="z")
                y = xr.DataArray(atl06sr.geometry.y.values, dims="z")
                sample = aligned_dem.interp(x=x, y=y)

                atl06sr["aligned_dem_height"] = sample.values
                atl06sr["icesat_minus_aligned_dem"] = (
                    atl06sr["h_mean"] - atl06sr["aligned_dem_height"]
                )
                self.atl06sr_processing_levels_filtered[key] = atl06sr

    def mapview_plot_atl06sr_to_dem(
        self,
        key="all",
        clim=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        if plot_aligned:
            column_name = "icesat_minus_aligned_dem"
            if not self.aligned_dem_fn:
                print("\nAligned DEM not found.\n")
                return
        else:
            column_name = "icesat_minus_dem"

        atl06sr = self.atl06sr_processing_levels_filtered[key]

        if column_name not in atl06sr.columns:
            print(
                f"\n{column_name} not found in ATL06 dataframe: {key}. Running differencing first.\n"
            )
            self.atl06sr_to_dem_dh()

        if clim is not None:
            symm_clim = False
        else:
            symm_clim = True

        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        epsg = dem.rio.crs.to_epsg()

        self.plot_atl06sr(
            key=key,
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
        self,
        key="all",
        title="Histogram",
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        def _nmad(a, c=1.4826):
            return np.nanmedian(np.fabs(a - np.nanmedian(a))) * c

        atl06sr = self.atl06sr_processing_levels_filtered[key]

        if "icesat_minus_dem" not in atl06sr.columns:
            print(
                f"\n'icesat_minus_dem' not found in ATL06 dataframe: {key}. Running differencing first.\n"
            )
            self.atl06sr_to_dem_dh()
            atl06sr = self.atl06sr_processing_levels_filtered[key]

        column_names = ["icesat_minus_dem"]
        if plot_aligned:
            column_names.append("icesat_minus_aligned_dem")
            if not self.aligned_dem_fn:
                print("\nAligned DEM not found.\n")
                return

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        for column_name in column_names:
            med = atl06sr[column_name].quantile(0.50)
            nmad = atl06sr[[column_name]].apply(_nmad).iloc[0]

            xmin = atl06sr[column_name].quantile(0.01)
            xmax = atl06sr[column_name].quantile(0.99)
            plot_kwargs = {"bins": 128, "alpha": 0.5, "range": (xmin, xmax)}
            atl06sr.hist(
                ax=ax,
                column=column_name,
                label=f"{column_name}, Median={med:0.2f}, NMAD={nmad:0.2f}",
                **plot_kwargs,
            )

        ax.legend()
        ax.set_title(None)
        ax.set_xlabel("ICESat-2 - DEM (m)")

        fig.suptitle(f"{title}\n{key} (n={atl06sr.shape[0]})", size=10)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    # TODO: https://github.com/uw-cryo/asp_plot/issues/40
    # def plot_atl06sr_dem_profiles(
    #     self,
    #     title="ICESat-2 ATL06-SR Profiles",
    #     select_years=None,
    #     select_months=None,
    #     select_days=None,
    #     only_strong_beams=True,
    #     save_dir=None,
    #     fig_fn=None,
    # ):
    #     if "icesat_minus_dem" not in self.atl06sr_filtered.columns:
    #         self.atl06sr_to_dem_dh()

    #     atl06sr = self.atl06sr_filtered

    #     # Additional day, month, and year filtering
    #     if select_years:
    #         atl06sr = atl06sr[atl06sr.index.year.isin(select_years)]
    #     if select_months:
    #         atl06sr = atl06sr[atl06sr.index.month.isin(select_months)]
    #     if select_days:
    #         atl06sr = atl06sr[atl06sr.index.day.isin(select_days)]

    #     # Get day of interest
    #     dates = atl06sr.index.strftime("%Y-%m-%d").unique()

    #     if dates.size > 1:
    #         logger.warning(
    #             f"\nYou are trying to plot {dates.size} ICESat-2 passes. Please apply additional day, month, and year filtering to get only one pass for plotting.\n"
    #         )
    #         return
    #     else:
    #         date = dates[0]

    #     atl06sr = atl06sr[atl06sr.index.normalize() == date]

    #     # Get unique beam strength spot numbers
    #     spots = atl06sr.spot.unique()

    #     # Optionally, filter out weak beams (2, 4, 6)
    #     if only_strong_beams:
    #         spots = spots[spots % 2 == 1]

    #     # Plot the beams
    #     fig, axes = plt.subplots(spots.size, 1, figsize=(10, 12))
    #     axes = axes.flatten()
    #     for ii, spot in enumerate(spots):
    #         ax = axes[ii]
    #         spot_to_plot = atl06sr[atl06sr.spot == spot]
    #         along_track_dist = abs(spot_to_plot.x_atc - spot_to_plot.x_atc.max()) / 1000

    #         ax.scatter(
    #             along_track_dist,
    #             spot_to_plot.h_mean,
    #             color="black",
    #             s=5,
    #             marker="s",
    #             label="ICESat-2 ATL06",
    #         )
    #         ax.scatter(
    #             along_track_dist,
    #             spot_to_plot.dem_aligned_height,
    #             color="red",
    #             s=5,
    #             marker="o",
    #             label="DEM",
    #         )
    #         ax.set_axisbelow(True)
    #         ax.grid(0.3)
    #         ax.set_title(f"Laser Spot {spot:0.0f}")
    #         ax.set_xlabel("Distance along track (km)")
    #         ax.set_ylabel("Elevation (m HAE)")
    #         ax.legend()

    #     fig.suptitle(title)
    #     fig.subplots_adjust(hspace=0.3)

    #     fig.tight_layout()
    #     if save_dir and fig_fn:
    #         save_figure(fig, save_dir, fig_fn)
