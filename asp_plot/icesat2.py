import os
import glob
import logging
import rioxarray
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sliderule import sliderule, icesat2

icesat2.init("slideruleearth.io", verbose=False)


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Icesat2:
    def __init__(self, dem_fn, geojson_fn):
        self.dem_fn = dem_fn
        self.geojson_fn = geojson_fn

    def pull_atl06_data(self, srt="land", cnf="high", ats=5, cnt=5, len=40, res=10, maxi=5, H_min_win=3, sigma_r_max=5):
        if not os.path.exists(self.geojson_fn):
            raise ValueError(f"Geojson file not found: {self.geojson_fn}\nUse this tool to make and download one:\nhttps://geojson.io/")

        region = sliderule.toregion(self.geojson_fn)["poly"]

        # TODO: make request

        # TODO: optionally save to parquet and/or csv

        return region

    # TODO: quick n' dirty plot, maybe with ctx basemap?
    # def plot_atl06_data(self, region):

    # TODO: clean it up
    # def clean_atl06_data(self):
        # TODO: optionally save to parquet and/or csv

    # TODO: comparison plotting

    # TODO: profile plotting

    
