import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors
import contextily as ctx

from asp_plot.utils import *
from asp_plot.io import *

def resid_plot(
    geodataframes,
    col="mean_residual",
    clip_final=True,
    lognorm=False,
    clim=None,
    cmap="inferno",
    map_crs="EPSG:4326",
    **ctx_kwargs,
):
    n = len(geodataframes)
    nrows = (n + 3) // 4
    ncols = min(n, 4)
    f, axa = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)
    axa = axa.flatten()
    for i, gdf in enumerate(geodataframes):
        if clim is None:
            clim = get_clim(gdf[col], perc=(0, 98))
        vmin, vmax = clim
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
        gdf.sort_values(by=col).to_crs(map_crs).plot(ax=axa[i], column=col, **plot_kw)
        ctx.add_basemap(ax=axa[i], **ctx_kwargs)
        if clip_final and i == n - 1:
            axa[i].autoscale(False)
        axa[i].set_title(f"{gdf.filename}\nResiduals (n={gdf.shape[0]})")
    for i in range(n, nrows * ncols):
        f.delaxes(axa[i])
    plt.tight_layout()

def resid_plot_single(
    residuals,
    col="mean_residual",
    lognorm=False,
    clim=None,
    cmap="inferno",
    map_crs="EPSG:4326",
    **ctx_kwargs,
):
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    if clim is None:
        clim = get_clim(residuals[col], perc=(0, 98))
    vmin, vmax = clim
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
    residuals.sort_values(by=col).to_crs(map_crs).plot(ax=ax, column=col, **plot_kw)
    ctx.add_basemap(ax=ax, **ctx_kwargs)
    ax.set_title(f"Residuals (n={residuals.shape[0]})")
    plt.tight_layout()


def resid_plot_before_after(
    init,
    final,
    col="mean_residual",
    clip_final=True,
    lognorm=False,
    clim=None,
    cmap="inferno",
    map_crs="EPSG:4326",
    **ctx_kwargs,
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
