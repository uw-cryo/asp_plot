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
from asp_plot.icesat2_source import (
    MAX_GRANULES_LISTED,
    Icesat2Source,
    SlideRuleRequestError,
)

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
    alt.icesat2._list_granules = lambda parms: []
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


class TestRequestReturnsNothing:
    """SlideRule's two ways of returning nothing: ``None`` when the client
    cannot open its output file, and a frame with columns but no rows when the
    request ran and found no points."""

    # The form Raster.get_bounds() gives the request when no region is passed.
    POLY = [
        {"lon": -149.9, "lat": 70.0},
        {"lon": -149.9, "lat": 70.1},
        {"lon": -149.8, "lat": 70.1},
        {"lon": -149.8, "lat": 70.0},
        {"lon": -149.9, "lat": 70.0},
    ]

    def _request(self, src, monkeypatch, result, region=POLY):
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.run", lambda api, parms: result
        )
        src.request_atl06sr_multi_processing(
            processing_levels=["canopy"],
            region=region,
            time_range="buffered",
            t0="2019-01-01",
            t1="2020-01-01",
        )

    def test_none_raises_error_naming_the_request(self, src, monkeypatch):
        with pytest.raises(RuntimeError) as excinfo:
            self._request(src, monkeypatch, None)
        message = str(excinfo.value)
        assert "SlideRule returned no result" in message
        assert "atl03x" in message
        assert "lon -149.9000 to -149.8000, lat 70.0000 to 70.1000" in message
        assert "2019-01-01" in message
        assert "2020-01-01" in message
        assert "--atl06sr-time-range" in message
        assert "NoneType" not in message

    def test_none_error_handles_a_region_given_as_bounds(self, src, monkeypatch):
        with pytest.raises(RuntimeError, match="SlideRule returned no result"):
            self._request(src, monkeypatch, None, region=REGION)

    def test_no_rows_warns_and_continues(self, src, monkeypatch, caplog):
        with caplog.at_level("WARNING", logger="asp_plot.icesat2_source"):
            self._request(src, monkeypatch, _synthetic_atl06sr().iloc[0:0])
        assert "SlideRule found no points" in caplog.text
        assert len(src.atl06sr_processing_levels["canopy"]) == 0
        assert len(src.atl06sr_processing_levels_filtered["canopy"]) == 0


class TestSlideRuleUnavailable:
    """A SlideRule outage: the startup check fails, or ``run()`` raises from
    inside the client once its retries are used up (#214)."""

    def test_run_raising_becomes_one_error_naming_the_request(self, src, monkeypatch):
        def fail_run(api, parms):
            raise TypeError("'NoneType' object is not iterable")

        monkeypatch.setattr("asp_plot.icesat2_source.sliderule_api.run", fail_run)
        with pytest.raises(SlideRuleRequestError) as excinfo:
            src.request_atl06sr_multi_processing(
                processing_levels=["all"],
                region=REGION,
                time_range="buffered",
                t0="2019-01-01",
                t1="2020-01-01",
            )
        message = str(excinfo.value)
        assert "SlideRule request failed" in message
        assert "atl03x request over [-149.9, 70.0, -149.8, 70.1]" in message
        assert "2019-01-01" in message
        assert "may succeed if repeated" in message
        assert isinstance(excinfo.value.__cause__, TypeError)

    def test_none_result_is_the_same_error(self, src, monkeypatch):
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.run", lambda api, parms: None
        )
        with pytest.raises(SlideRuleRequestError, match="returned no result"):
            src.request_atl06sr_multi_processing(
                processing_levels=["all"], region=REGION
            )

    def test_failed_startup_check_sends_no_request(self, src, monkeypatch):
        src._sliderule_initialized = False
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.init", lambda *a, **k: False
        )

        def unexpected_run(api, parms):
            raise AssertionError("run() called after a failed startup check")

        monkeypatch.setattr("asp_plot.icesat2_source.sliderule_api.run", unexpected_run)
        with pytest.raises(SlideRuleRequestError, match="Could not connect"):
            src.request_atl06sr_multi_processing(
                processing_levels=["all"], region=REGION
            )
        # The next request tries to connect again.
        assert src._sliderule_initialized is False

    def test_cache_hit_does_not_need_sliderule(self, src, monkeypatch):
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.run",
            lambda api, parms: _synthetic_atl06sr(),
        )
        src.request_atl06sr_multi_processing(
            processing_levels=["all"], region=REGION, save_to_parquet=True
        )

        src._sliderule_initialized = False
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.init", lambda *a, **k: False
        )
        src.request_atl06sr_multi_processing(
            processing_levels=["all"], region=REGION, save_to_parquet=True
        )
        assert len(src.atl06sr_processing_levels["all"]) > 0


class TestGranuleBatches:
    """A request over the server's granule limit is split by granule, and the
    parquet cache sees only the overall request."""

    GRANULES = [f"ATL03_{i:03d}.h5" for i in range(5)]

    @pytest.fixture
    def calls(self, src, monkeypatch):
        calls = []

        def fake_run(api, parms):
            calls.append(parms.get("resources"))
            return _synthetic_atl06sr(n=2)

        monkeypatch.setattr("asp_plot.icesat2_source.sliderule_api.run", fake_run)
        monkeypatch.setattr("asp_plot.icesat2_source.MAX_GRANULES_PER_REQUEST", 2)
        src._list_granules = lambda parms: self.GRANULES
        return calls

    def _request(self, src):
        src.request_atl06sr_multi_processing(
            processing_levels=["all"],
            region=REGION,
            save_to_parquet=True,
            filename="atl06sr",
        )

    def test_each_granule_in_one_batch(self, src, calls):
        self._request(src)
        assert calls == [self.GRANULES[:2], self.GRANULES[2:4], self.GRANULES[4:]]

    def test_results_are_combined(self, src, calls):
        self._request(src)
        assert isinstance(src.atl06sr_processing_levels["all"], gpd.GeoDataFrame)
        assert len(src.atl06sr_processing_levels["all"]) == 6

    def test_repeat_is_a_cache_hit(self, src, calls):
        self._request(src)
        assert len(calls) == 3
        self._request(src)
        assert len(calls) == 3, "cache hit should not trigger a new request"

    def test_batch_with_no_points_is_left_out(self, src, calls, monkeypatch):
        results = iter(
            [
                _synthetic_atl06sr().iloc[0:0],
                _synthetic_atl06sr(n=2),
                _synthetic_atl06sr(n=2),
            ]
        )
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.run",
            lambda api, parms: next(results),
        )
        self._request(src)
        assert len(src.atl06sr_processing_levels["all"]) == 4

    def test_all_batches_empty_warns(self, src, calls, monkeypatch, caplog):
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.run",
            lambda api, parms: _synthetic_atl06sr().iloc[0:0],
        )
        with caplog.at_level("WARNING", logger="asp_plot.icesat2_source"):
            self._request(src)
        assert "SlideRule found no points" in caplog.text

    def test_failed_batch_is_named(self, src, calls, monkeypatch):
        results = iter([_synthetic_atl06sr(n=2), None])
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_api.run",
            lambda api, parms: next(results),
        )
        with pytest.raises(RuntimeError, match="batch 2 of 3"):
            self._request(src)

    def test_at_the_limit_is_one_request(self, src, calls):
        src._list_granules = lambda parms: self.GRANULES[:2]
        self._request(src)
        assert calls == [None]


class TestListGranules:
    PARMS = {"poly": REGION, "t0": "2019-01-01", "t1": "2020-01-01"}

    def test_listing_asks_for_more_than_the_request_limit(self, monkeypatch):
        # The server applies the 300-granule limit to the listing as well,
        # unless the query asks for more: without this, the listing fails on
        # exactly the requests that need batching.
        limits = []
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_earthdata.set_max_resources",
            limits.append,
        )
        monkeypatch.setattr(
            "asp_plot.icesat2_source.sliderule_earthdata.cmr", lambda **kwargs: []
        )
        Icesat2Source._list_granules(self.PARMS)
        assert limits == [MAX_GRANULES_LISTED]
        assert MAX_GRANULES_LISTED > 300

    def test_sorted_and_scoped_to_the_request(self, monkeypatch):
        seen = {}

        def fake_cmr(**kwargs):
            seen.update(kwargs)
            return ["ATL03_2.h5", "ATL03_1.h5"]

        monkeypatch.setattr("asp_plot.icesat2_source.sliderule_earthdata.cmr", fake_cmr)
        assert Icesat2Source._list_granules(self.PARMS) == [
            "ATL03_1.h5",
            "ATL03_2.h5",
        ]
        assert seen == {
            "short_name": "ATL03",
            "polygon": REGION,
            "time_start": "2019-01-01",
            "time_end": "2020-01-01",
        }

    def test_failed_search_means_one_request(self, monkeypatch, caplog):
        def fail_cmr(**kwargs):
            raise OSError("CMR is down")

        monkeypatch.setattr("asp_plot.icesat2_source.sliderule_earthdata.cmr", fail_cmr)
        with caplog.at_level("WARNING", logger="asp_plot.icesat2_source"):
            assert Icesat2Source._list_granules(self.PARMS) == []
        assert "CMR is down" in caplog.text


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
