"""Tests for the shared AltimetrySource base machinery.

The DEM-sampling, std-outlier-mask and CSV-writer helpers were lifted out of
the ICESat-2 and planetary sources (#140); these exercise them directly.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from asp_plot.altimetry_source import AltimetrySource
from asp_plot.utils import Raster

DEM_FN = "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"


class TestStdOutlierMask:
    def test_keeps_within_sigma_drops_outlier_keeps_nan(self):
        dh = pd.Series([0.0, 0.1, -0.1, 0.05, 100.0, np.nan])
        mask = AltimetrySource._std_outlier_mask(dh, n_sigma=1)
        assert mask is not None
        assert mask.iloc[0]  # central value kept
        assert not mask.iloc[4]  # extreme outlier dropped
        assert mask.iloc[5]  # NaN row kept (no dh yet)

    def test_degenerate_std_returns_none(self):
        # Zero spread → no meaningful threshold → signal "do not filter".
        assert AltimetrySource._std_outlier_mask(pd.Series([5.0, 5.0, 5.0]), 3) is None

    def test_all_nan_returns_none(self):
        assert AltimetrySource._std_outlier_mask(pd.Series([np.nan, np.nan]), 3) is None


class TestInterpDemAtPoints:
    def test_returns_values_and_reprojected_points(self):
        raster = Raster(DEM_FN)
        lon_min, lat_min, lon_max, lat_max = raster.get_bounds(
            latlon=True, json_format=False
        )
        pts = gpd.GeoDataFrame(
            geometry=[Point((lon_min + lon_max) / 2, (lat_min + lat_max) / 2)],
            crs="EPSG:4326",
        )
        dem = AltimetrySource._open_dem(DEM_FN)
        sampled, reproj = AltimetrySource._interp_dem_at_points(dem, pts)
        assert len(sampled) == 1
        # Points come back in the DEM CRS (the working CRS downstream code uses).
        assert reproj.crs == dem.rio.crs
        # A point at the DEM centre should sample a finite elevation.
        assert np.isfinite(sampled[0])


class TestWriteCsvToDirectory:
    def test_writes_under_coordinator_directory(self, tmp_path):
        class FakeAlt:
            directory = str(tmp_path)

        src = AltimetrySource(FakeAlt())
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = src._write_csv_to_directory(df, "out.csv")
        assert path == str(tmp_path / "out.csv")
        back = pd.read_csv(path)
        assert list(back.columns) == ["a", "b"]
        assert len(back) == 2
