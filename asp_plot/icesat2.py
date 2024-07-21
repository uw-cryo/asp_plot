import logging
import os

import geopandas as gpd
from sliderule import icesat2, sliderule

icesat2.init("slideruleearth.io", verbose=True)


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ICESat2:
    def __init__(self, dem_fn, geojson_fn, atl06=None):
        self.dem_fn = dem_fn
        self.geojson_fn = geojson_fn
        if atl06 is not None and not isinstance(atl06, gpd.GeoDataFrame):
            raise ValueError("atl06 must be a GeoDataFrame if provided.")
        self.atl06 = atl06

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

    # TODO: clean it up
    # def clean_atl06_data(self):
    # TODO: optionally save to parquet and/or csv
    # parquet needs time in [ms] so some precision loss
    # atl06.index = atl06.index.astype("datetime64[ms]")
    # csv will only save lat/lon/height (and time if possible?)

    # def plot_atl06_data(self, region):

    # TODO: comparison plotting

    # TODO: profile plotting
