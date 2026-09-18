"""Tests for the shared AltimetrySource base machinery.

The DEM-sampling, outlier-mask and CSV-writer helpers were lifted out of
the ICESat-2 and planetary sources (#140); these exercise them directly.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from asp_plot.altimetry_source import AltimetrySource
from asp_plot.utils import Raster

DEM_FN = "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"


class TestOutlierMask:
    def test_keeps_within_sigma_drops_outlier_keeps_nan(self):
        dh = pd.Series([0.0, 0.1, -0.1, 0.05, 100.0, np.nan])
        mask = AltimetrySource._outlier_mask(dh, n_sigma=1)
        assert mask is not None
        assert mask.iloc[0]  # central value kept
        assert not mask.iloc[4]  # extreme outlier dropped
        assert mask.iloc[5]  # NaN row kept (no dh yet)

    def test_survives_a_large_cluster_of_gross_outliers(self):
        # One cloud-contaminated ICESat-2 pass: a fifth of the sample sits
        # 150-200 m above the ground. A plain mean/std cut keeps all of it
        # (the cluster inflates the std to ~70 m); the gross cut at 30 NMAD
        # removes it, and the 3-std cut then runs on the ground points alone.
        rng = np.random.default_rng(0)
        ground = rng.normal(0.0, 1.0, 800)
        cloud = rng.normal(175.0, 15.0, 200)
        dh = pd.Series(np.concatenate([ground, cloud]))
        mask = AltimetrySource._outlier_mask(dh, n_sigma=3)
        assert not mask.iloc[800:].any()
        assert mask.iloc[:800].mean() > 0.99
        # The old cut, for the record
        std_mask = (dh - dh.mean()).abs() <= 3 * dh.std()
        assert std_mask.all()

    def test_heavy_tails_are_still_cut_by_std_not_nmad(self):
        # A DSM against ICESat-2 over a city has real residuals of many NMADs
        # (building edges, trees); those are DEM error, not blunders, and the
        # cut must stay the mean/std one so existing report numbers hold.
        rng = np.random.default_rng(1)
        core = rng.normal(0.0, 0.5, 900)
        tails = rng.uniform(-8.0, 8.0, 100)  # up to ~16 NMAD, under the 30 gate
        dh = pd.Series(np.concatenate([core, tails]))
        mask = AltimetrySource._outlier_mask(dh, n_sigma=3)
        std_mask = (dh - dh.mean()).abs() <= 3 * dh.std()
        pd.testing.assert_series_equal(mask, std_mask)

    def test_degenerate_spread_returns_none(self):
        # Zero spread → no meaningful threshold → signal "do not filter".
        assert AltimetrySource._outlier_mask(pd.Series([5.0, 5.0, 5.0]), 3) is None

    def test_all_nan_returns_none(self):
        assert AltimetrySource._outlier_mask(pd.Series([np.nan, np.nan]), 3) is None


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
