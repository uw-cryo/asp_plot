import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray
from sliderule import icesat2, sliderule

from asp_plot.utils import ColorBar, Plotter, save_figure

icesat2.init("slideruleearth.io", verbose=True)


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ICESat2(Plotter):
    def __init__(self, dem_fn, geojson_fn, atl06=None, **kwargs):
        super().__init__(**kwargs)
        self.dem_fn = dem_fn
        self.geojson_fn = geojson_fn
        if atl06 is not None and not isinstance(atl06, gpd.GeoDataFrame):
            raise ValueError("atl06 must be a GeoDataFrame if provided.")
        self.atl06 = atl06
        self.atl06_clean = None

    def pull_atl06_data(
        self,
        esa_worldcover=True,
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

        return self.atl06

    def clean_atl06(self, h_sigma_quantile=0.95, mask_worldcover_water=True):
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

        return self.atl06_clean

    def plot_atl06(
        self,
        clean=False,
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
            atl06_sorted = self.atl06_clean.sort_values(by=column_name).to_crs(map_crs)
        else:
            atl06_sorted = self.atl06.sort_values(by=column_name).to_crs(map_crs)

        if clim is None:
            clim = ColorBar().get_clim(self.atl06[column_name])

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        self.plot_geodataframe(
            ax=ax,
            gdf=atl06_sorted,
            column_name=column_name,
            cbar_label=cbar_label,
            cmap=cmap,
            **ctx_kwargs,
        )

        fig.suptitle(self.title, size=10)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    # TODO: comparison plotting

    # TODO: profile plotting
