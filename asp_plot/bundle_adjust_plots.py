import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors
import contextily as ctx

from asp_plot.utils import *


def read_residuals(csv_fn):
    resid_cols = [
        "lon",
        "lat",
        "height_above_datum",
        "mean_residual",
        "num_observations",
    ]
    resid_df = pd.read_csv(csv_fn, skiprows=2, names=resid_cols)
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
        geometry=gpd.points_from_xy(resid_df["lon"], resid_df["lat"], crs="EPSG:4326"),
    )
    return resid_gdf


def resid_plot(
    init,
    final,
    col="mean_residual",
    clip_final=True,
    lognorm=False,
    clim=None,
    cmap="inferno",
    map_crs="EPSG:4326",
    **ctx_kwargs
):
    f, axa = plt.subplots(1, 2, figsize=(10, 3), sharex=True, sharey=True)
    if clim is None:
        clim_init = get_clim(init[col], perc=(0, 98))
        clim_final = get_clim(final[col], perc=(0, 98))
        vmin = min(clim_init[0], clim_final[0])
        vmax = max(clim_init[1], clim_final[1])
    else:
        vmin, vmax = clim
    print(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if lognorm:
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    plot_kw = {
        "cmap": cmap,
        "norm": norm,
        "s": 1,
        "legend": True,
        "legend_kwds": {"label": col},
    }
    final.sort_values(by=col).to_crs(map_crs).plot(ax=axa[1], column=col, **plot_kw)
    ctx.add_basemap(ax=axa[1], **ctx_kwargs)
    if clip_final:
        axa[0].autoscale(False)
    init.sort_values(by=col).to_crs(map_crs).plot(ax=axa[0], column=col, **plot_kw)
    ctx.add_basemap(ax=axa[0], **ctx_kwargs)
    axa[0].set_title(f"Initial Residuals (n={init.shape[0]})")
    axa[1].set_title(f"Final Residuals (n={final.shape[0]})")
    plt.tight_layout()
