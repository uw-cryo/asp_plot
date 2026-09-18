"""Tests for the decomposed ICESat-2 ATL06-SR request path.

``request_atl06sr_multi_processing`` is network-bound (it calls SlideRule), so
before #140 its cache-parameter-comparison logic was never exercised in CI.
These tests mock the SlideRule request with a small synthetic ATL06-SR
GeoDataFrame so the decomposed helpers — ``_build_level_parms``,
``_params_match_cache``, ``_load_cached_atl06sr`` and ``_request_atl06sr_level``
— and the parquet cache round-trip are covered without a network call.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from asp_plot.altimetry import Altimetry
from asp_plot.icesat2_source import Icesat2Source

DEM_FN = "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"
REGION = [-149.90, 70.00, -149.80, 70.10]


def _synthetic_atl06sr(n=6):
    """A minimal ATL06-SR-like GeoDataFrame the ingest path accepts.

    Carries the columns ``_ingest_atl06sr`` reads (``cycle``, ``h_sigma``) plus
    a tz-naive DatetimeIndex named ``time``, mimicking a SlideRule result.
    """
    times = pd.date_range("2022-04-01", periods=n, freq="D")
    lons = np.linspace(REGION[0], REGION[2], n)
    lats = np.linspace(REGION[1], REGION[3], n)
    gdf = gpd.GeoDataFrame(
        {
            "cycle": np.full(n, 5),
            "h_sigma": np.linspace(0.1, 0.5, n),
            "h_mean": np.linspace(100.0, 110.0, n),
            "rgt": np.full(n, 1),
            "spot": np.full(n, 1),
            "x_atc": np.linspace(0.0, 1000.0, n),
        },
        geometry=[Point(x, y) for x, y in zip(lons, lats)],
        crs="EPSG:4326",
    )
    gdf.index = pd.DatetimeIndex(times, name="time")
    return gdf


@pytest.fixture
def src(tmp_path):
    """An Icesat2Source whose coordinator writes into a temp directory."""
    alt = Altimetry(directory=str(tmp_path), dem_fn=DEM_FN)
    # Avoid the real SlideRule session init and WorldCover S3 sampling.
    alt.icesat2._sliderule_initialized = True
    alt.icesat2._sample_worldcover_into_gdf = lambda df: df
    return alt.icesat2


class TestBuildLevelParms:
    def test_only_requested_levels_returned(self):
        parms = Icesat2Source._build_level_parms(
            ["all", "ground"], REGION, "t0", "t1", 20, 40, 20, 6, 10
        )
        assert set(parms.keys()) == {"all", "ground"}

    def test_shared_and_custom_merged(self):
        parms = Icesat2Source._build_level_parms(
            ["all"], REGION, "t0", "t1", 20, 40, 20, 6, 10
        )["all"]
        # Shared envelope
        assert parms["poly"] == REGION
        assert parms["t0"] == "t0" and parms["t1"] == "t1"
        assert parms["res"] == 20 and parms["len"] == 40 and parms["ats"] == 20
        assert parms["fit"] == {"maxi": 6}
        # Level-specific
        assert parms["cnf"] == "atl03_high"
        assert parms["srt"] == -1
        assert parms["cnt"] == 10

    def test_ground_uses_reduced_count_and_class(self):
        parms = Icesat2Source._build_level_parms(
            ["ground"], REGION, "t0", "t1", 20, 40, 20, 6, 10
        )["ground"]
        assert parms["cnt"] == 5
        assert parms["atl08_class"] == "atl08_ground"


class TestParamsMatchCache:
    def test_matches_after_round_trip(self, src, tmp_path):
        parms = Icesat2Source._build_level_parms(
            ["all"], REGION, "t0", "t1", 20, 40, 20, 6, 10
        )["all"]
        fn = str(tmp_path / "atl06sr_all.parquet")
        src._save_to_parquet(fn, _synthetic_atl06sr(), parms)
        cached = gpd.read_parquet(fn)
        assert src._params_match_cache(parms, cached) is True

    def test_mismatch_when_param_differs(self, src, tmp_path):
        saved = Icesat2Source._build_level_parms(
            ["all"], REGION, "t0", "t1", 20, 40, 20, 6, 10
        )["all"]
        fn = str(tmp_path / "atl06sr_all.parquet")
        src._save_to_parquet(fn, _synthetic_atl06sr(), saved)
        cached = gpd.read_parquet(fn)
        # Different resolution → should not be recognized as a cache hit.
        other = Icesat2Source._build_level_parms(
            ["all"], REGION, "t0", "t1", 30, 40, 20, 6, 10
        )["all"]
        assert src._params_match_cache(other, cached) is False

    def test_no_parameters_column(self, src):
        assert src._params_match_cache({"res": 20}, _synthetic_atl06sr()) is False


class TestRequestAtl06srMultiProcessing:
    """End-to-end through the orchestrator with SlideRule mocked."""

    def _patch_run(self, monkeypatch):
        calls = {"n": 0}

        def fake_run(api, parms):
            calls["n"] += 1
            return _synthetic_atl06sr()

        monkeypatch.setattr("asp_plot.icesat2_source.sliderule_api.run", fake_run)
        return calls

    def test_fresh_request_then_cache_hit(self, src, monkeypatch):
        calls = self._patch_run(monkeypatch)

        src.request_atl06sr_multi_processing(
            processing_levels=["all"],
            region=REGION,
            save_to_parquet=True,
            filename="atl06sr",
        )
        assert calls["n"] == 1
        assert "all" in src.atl06sr_processing_levels
        assert "all" in src.atl06sr_processing_levels_filtered
        # cycle >= 3 retained, h_sigma != 0 retained
        assert len(src.atl06sr_processing_levels_filtered["all"]) > 0

        # Second call with identical params: served from the parquet cache.
        src.request_atl06sr_multi_processing(
            processing_levels=["all"],
            region=REGION,
            save_to_parquet=True,
            filename="atl06sr",
        )
        assert calls["n"] == 1, "cache hit should not trigger a new request"

    def test_param_change_regenerates(self, src, monkeypatch):
        calls = self._patch_run(monkeypatch)

        src.request_atl06sr_multi_processing(
            processing_levels=["all"], region=REGION, save_to_parquet=True, res=20
        )
        assert calls["n"] == 1

        # Different resolution → cached params no longer match → re-request.
        src.request_atl06sr_multi_processing(
            processing_levels=["all"], region=REGION, save_to_parquet=True, res=30
        )
        assert calls["n"] == 2

    def test_records_request_metadata(self, src, monkeypatch):
        self._patch_run(monkeypatch)
        src.request_atl06sr_multi_processing(
            processing_levels=["all", "ground"], region=REGION, save_to_parquet=False
        )
        assert src.atl06sr_request_parms["processing_levels"] == ["all", "ground"]
        assert set(src.atl06sr_parquet_paths.keys()) == {"all", "ground"}


class TestAlignedResidualsShareThePointsOfTheUnaligned:
    """A translated DEM's holes move; the aligned residuals must not gain points.

    Two 10 x 10 m rasters at 1 m: the "unaligned" DEM has a NaN hole under one
    point, the "aligned" copy has its hole one pixel over, so that point is
    NaN before alignment and valid after. Its ICESat-2 height is a cloud
    return 200 m up, which the outlier cut never saw because NaN rows pass it.
    """

    @staticmethod
    def _write(fn, z):
        import rasterio
        from rasterio.transform import from_origin

        with rasterio.open(
            fn,
            "w",
            driver="GTiff",
            height=z.shape[0],
            width=z.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:32611",
            transform=from_origin(0, 10, 1, 1),
            nodata=-9999,
        ) as dst:
            dst.write(np.where(np.isnan(z), -9999, z).astype("float32"), 1)

    def test_aligned_dh_is_nan_where_unaligned_dh_is_nan(self, tmp_path):
        z = np.full((10, 10), 100.0)
        hole_before = z.copy()
        hole_before[4:6, 4:6] = np.nan  # covers the point at (4.5, 5.5)
        hole_after = z.copy()
        hole_after[4:6, 7:9] = np.nan  # hole moved; the point now samples 100 m
        self._write(tmp_path / "dem.tif", hole_before)
        self._write(tmp_path / "dem_aligned.tif", hole_after)

        pts = gpd.GeoDataFrame(
            {"h_mean": [100.2, 99.9, 300.0]},
            geometry=[Point(1.5, 8.5), Point(2.5, 8.5), Point(4.5, 5.5)],
            crs="EPSG:32611",
        )
        alt = Altimetry(
            directory=str(tmp_path),
            dem_fn=str(tmp_path / "dem.tif"),
            aligned_dem_fn=str(tmp_path / "dem_aligned.tif"),
            atl06sr_processing_levels={"all": pts},
            atl06sr_processing_levels_filtered={"all": pts.copy()},
        )
        alt.atl06sr_to_dem_dh(n_sigma=None)
        got = alt.atl06sr_processing_levels_filtered["all"]

        assert np.isnan(got["icesat_minus_dem"].iloc[2])
        assert np.isfinite(got["aligned_dem_height"].iloc[2])  # it did sample
        assert np.isnan(got["icesat_minus_aligned_dem"].iloc[2])  # but is not scored
        assert np.allclose(got["icesat_minus_aligned_dem"].iloc[:2], [0.2, -0.1])
