from datetime import datetime, timezone

import geopandas as gpd
import matplotlib
import pytest

from asp_plot.altimetry import ICESAT2_MISSION_START, Altimetry

matplotlib.use("Agg")


class TestAltimetry:
    @pytest.fixture
    def icesat(self):
        icesat = Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

        # The actual request is fairly time consuming, thus better to be skipped for testing.
        # icesat.request_atl06sr_multi_processing(
        #     filename="tests/test_data/icesat_data/atl06sr_",
        # )
        for key in ["all", "ground", "canopy", "top_of_canopy"]:
            icesat.atl06sr_processing_levels[key] = gpd.read_parquet(
                f"tests/test_data/icesat_data/atl06sr_{key}.parquet"
            )
            icesat.atl06sr_processing_levels_filtered[key] = (
                icesat.atl06sr_processing_levels[key]
            )
        return icesat

    def test_filter_esa_worldcover(self, icesat):
        try:
            icesat.filter_esa_worldcover()
        except Exception as e:
            pytest.fail(f"filter_esa_worldcover() method raised an exception: {str(e)}")

    def test_generic_temporal_filter_atl06sr(self, icesat):
        try:
            icesat.generic_temporal_filter_atl06sr(
                select_years=[2019, 2020],
                select_months=[1, 2, 3],
                select_days=[1, 2, 3],
            )
        except Exception as e:
            pytest.fail(
                f"generic_temporal_filter_atl06sr() method raised an exception: {str(e)}"
            )

    def test_predefined_temporal_filter_atl06sr(self, icesat):
        try:
            icesat.predefined_temporal_filter_atl06sr()
        except Exception as e:
            pytest.fail(
                f"predefined_temporal_filter_atl06sr() method raised an exception: {str(e)}"
            )

    def test_plot_atl06sr_time_stamps(self, icesat):
        try:
            icesat.plot_atl06sr_time_stamps()
        except Exception as e:
            pytest.fail(
                f"plot_atl06sr_time_stamps() method raised an exception: {str(e)}"
            )

    def test_plot_atl06sr(self, icesat):
        try:
            icesat.plot_atl06sr()
        except Exception as e:
            pytest.fail(f"plot_atl06sr() method raised an exception: {str(e)}")

    def test_atl06sr_to_dem_dh(self, icesat):
        try:
            icesat.atl06sr_to_dem_dh()
        except Exception as e:
            pytest.fail(f"atl06sr_to_dem_dh() method raised an exception: {str(e)}")

    def test_mapview_plot_atl06sr_to_dem(self, icesat):
        try:
            icesat.mapview_plot_atl06sr_to_dem()
        except Exception as e:
            pytest.fail(
                f"mapview_plot_atl06sr_to_dem() method raised an exception: {str(e)}"
            )

    def test_histogram(self, icesat):
        try:
            icesat.histogram(key="ground_45_day_pad")
        except Exception as e:
            pytest.fail(f"histogram() method raised an exception: {str(e)}")

    def test_select_best_track(self, icesat):
        icesat.atl06sr_to_dem_dh()
        result = icesat._select_best_track(key="all")
        assert result is not None
        assert "rgt" in result
        assert "cycle" in result
        assert "spot" in result
        assert "count" in result
        assert "date" in result
        assert result["count"] > 0

    def test_histogram_by_landcover(self, icesat):
        try:
            icesat.histogram_by_landcover(key="all")
        except Exception as e:
            pytest.fail(
                f"histogram_by_landcover() method raised an exception: {str(e)}"
            )

    def test_plot_atl06sr_dem_profile(self, icesat):
        try:
            icesat.plot_atl06sr_dem_profile(key="all")
        except Exception as e:
            pytest.fail(
                f"plot_atl06sr_dem_profile() method raised an exception: {str(e)}"
            )

    def test_alignment_report(self, icesat):
        try:
            icesat.alignment_report()
        except Exception as e:
            pytest.fail(f"alignment_report() method raised an exception: {str(e)}")


class TestResolveTimeRange:
    @pytest.fixture
    def alt(self):
        return Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

    def test_explicit_date(self, alt):
        t0, t1, resolved = alt._resolve_time_range(scene_date="2022-06-15")
        assert resolved == datetime(2022, 6, 15, tzinfo=timezone.utc)
        assert t0 == "2021-06-15T00:00:00Z"
        assert t1 == "2023-06-15T00:00:00Z"

    def test_t0_clamped_to_mission_start(self, alt):
        t0, t1, resolved = alt._resolve_time_range(scene_date="2019-03-01")
        assert resolved == datetime(2019, 3, 1, tzinfo=timezone.utc)
        assert t0 == ICESAT2_MISSION_START.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert t1 == "2020-02-29T00:00:00Z"

    def test_pre_mission_date_triggers_fallback(self, alt):
        t0, t1, resolved = alt._resolve_time_range(scene_date="2015-01-01")
        assert resolved is None
        now = datetime.now(tz=timezone.utc)
        assert t0 >= ICESAT2_MISSION_START.strftime("%Y-%m-%dT%H:%M:%SZ")
        t1_dt = datetime.strptime(t1, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        assert abs((t1_dt - now).total_seconds()) < 60

    def test_auto_detect_from_xml(self, alt):
        t0, t1, resolved = alt._resolve_time_range()
        # Test data XMLs have dates around 2022-04-19
        assert resolved is not None
        assert resolved.year == 2022
        assert resolved.month == 4

    def test_custom_buffer(self, alt):
        t0, t1, resolved = alt._resolve_time_range(
            scene_date="2022-06-15", time_buffer_days=30
        )
        assert t0 == "2022-05-16T00:00:00Z"
        assert t1 == "2022-07-15T00:00:00Z"

    def test_cached_attributes(self, alt):
        alt._resolve_time_range(scene_date="2022-06-15")
        assert alt._scene_date == datetime(2022, 6, 15, tzinfo=timezone.utc)
        assert alt._t0 == datetime(2021, 6, 15, tzinfo=timezone.utc)
        assert alt._t1 == datetime(2023, 6, 15, tzinfo=timezone.utc)
