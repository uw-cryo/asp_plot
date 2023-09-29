import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors
import contextily as ctx

from asp_plot import utils

def read_residuals(csv_fn):
    resid_cols=['lon', 'lat', 'height_above_datum', 'mean_residual', 'num_observations']
    resid_df = pd.read_csv(csv_fn, skiprows=2, names=resid_cols)
    #Need the astype('str') to handle cases where column has dtype of int (without the # from DEM appended to some rows)
    resid_df['from_DEM'] = resid_df['num_observations'].astype('str').str.contains('# from DEM')
    resid_df['num_observations'] = resid_df['num_observations'].astype('str').str.split('#', expand=True)[0].astype(int)
    resid_gdf = gpd.GeoDataFrame(resid_df, geometry=gpd.points_from_xy(resid_df['lon'], resid_df['lat'], crs='EPSG:4326'))
    return resid_gdf
