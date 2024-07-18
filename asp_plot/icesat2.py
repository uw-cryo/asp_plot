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

    def pull_sliderule_data(self):
        if not os.path.exists(self.geojson_fn):
            raise ValueError(f"Geojson file not found: {self.geojson_fn}\nUse this tool to make and download one:\nhttps://geojson.io/")

        region = sliderule.toregion(self.geojson_fn)["poly"]

        return region
