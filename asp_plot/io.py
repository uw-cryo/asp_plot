import os
import pandas as pd
import geopandas as gpd


def get_residual_csv_paths(outdir, ba_prefix, two_stage=False):
    if two_stage:
        filenames = [
        f"{ba_prefix}-final_residuals_pointmap.csv",
        f"{ba_prefix}_pc_align-final_residuals_pointmap.csv",
    ]
    else:
        filenames = [
        f"{ba_prefix}-initial_residuals_pointmap.csv",
        f"{ba_prefix}-final_residuals_pointmap.csv",
    ]
        
    paths = [os.path.join(os.path.expanduser(outdir), filename) for filename in filenames]

    for path in paths:
        if not os.path.isfile(path):
            raise ValueError(f"Residuals CSV file not found: {path}")
    
    init_resid_path, final_resid_path = paths

    return init_resid_path, final_resid_path


def read_residuals_csv(csv_path):
    resid_cols = [
        "lon",
        "lat",
        "height_above_datum",
        "mean_residual",
        "num_observations",
    ]

    resid_df = pd.read_csv(csv_path, skiprows=2, names=resid_cols)

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

    resid_gdf.filename = os.path.basename(csv_path)

    return resid_gdf


def get_residual_gdfs(outdir, ba_prefix, two_stage=False):
    init_resid_path, final_resid_path = get_residual_csv_paths(
        outdir, ba_prefix, two_stage
    )
    init_resid_gdf = read_residuals_csv(init_resid_path)
    final_resid_gdf = read_residuals_csv(final_resid_path)
    return init_resid_gdf, final_resid_gdf
