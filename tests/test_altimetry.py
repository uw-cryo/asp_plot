import os
from datetime import datetime, timezone

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from asp_plot.altimetry import ICESAT2_MISSION_START, AlignmentResult, Altimetry

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

    def test_filter_outliers(self, icesat):
        icesat.atl06sr_to_dem_dh()
        n_before = len(icesat.atl06sr_processing_levels_filtered["all"])
        icesat.filter_outliers(n_sigma=3)
        n_after = len(icesat.atl06sr_processing_levels_filtered["all"])
        assert n_after <= n_before

    def test_plot_atl06sr_dem_profile(self, icesat):
        try:
            icesat.plot_atl06sr_dem_profile(key="all")
        except Exception as e:
            pytest.fail(
                f"plot_atl06sr_dem_profile() method raised an exception: {str(e)}"
            )

    def test_plot_best_worst_segments(self, icesat):
        try:
            icesat.plot_best_worst_segments(key="all")
        except Exception as e:
            pytest.fail(
                f"plot_best_worst_segments() method raised an exception: {str(e)}"
            )

    def test_alignment_report(self, icesat):
        try:
            icesat.alignment_report()
        except Exception as e:
            pytest.fail(f"alignment_report() method raised an exception: {str(e)}")

    def test_histogram_by_landcover_plot_aligned_no_aligned_dem(self, icesat):
        """plot_aligned=True without an aligned DEM should warn but not raise."""
        icesat.atl06sr_to_dem_dh()
        try:
            icesat.histogram_by_landcover(key="all", plot_aligned=True)
        except Exception as e:
            pytest.fail(f"histogram_by_landcover(plot_aligned=True) raised: {str(e)}")

    def test_plot_best_worst_segments_plot_aligned_no_aligned_dem(self, icesat):
        """plot_aligned=True without an aligned DEM should warn but not raise."""
        icesat.atl06sr_to_dem_dh()
        try:
            icesat.plot_best_worst_segments(key="all", plot_aligned=True)
        except Exception as e:
            pytest.fail(f"plot_best_worst_segments(plot_aligned=True) raised: {str(e)}")

    def test_align_and_evaluate_insufficient_points(self, icesat, monkeypatch):
        """Empty alignment_report_df maps to the insufficient_points branch."""

        def fake_alignment_report(self, **kwargs):
            self.alignment_report_df = pd.DataFrame()

        monkeypatch.setattr(Altimetry, "alignment_report", fake_alignment_report)
        result = icesat.align_and_evaluate()
        assert isinstance(result, AlignmentResult)
        assert result.status == "insufficient_points"
        assert result.aligned_dem_fn is None
        assert icesat.aligned_dem_fn is None
        assert result.improvement_pct is None
        assert "skipped" in result.message.lower()

    def test_align_and_evaluate_no_improvement(self, icesat, monkeypatch, tmp_path):
        """Unchanged p50 triggers no_improvement and deletes the aligned DEM."""
        fake_aligned = tmp_path / "fake-DEM_pc_align_translated.tif"
        fake_aligned.write_bytes(b"dummy")

        def fake_alignment_report(self, **kwargs):
            self.aligned_dem_fn = str(fake_aligned)
            self.alignment_report_df = pd.DataFrame(
                [
                    {
                        "key": "all",
                        "p16_beg": 1.0,
                        "p50_beg": 2.0,
                        "p84_beg": 3.0,
                        "p16_end": 1.0,
                        "p50_end": 2.0,
                        "p84_end": 3.0,
                        "north_shift": 0.0,
                        "east_shift": 0.0,
                        "down_shift": 0.0,
                        "translation_magnitude": 0.0,
                    }
                ]
            )

        monkeypatch.setattr(Altimetry, "alignment_report", fake_alignment_report)
        result = icesat.align_and_evaluate(improvement_threshold_pct=5.0)
        assert result.status == "no_improvement"
        assert result.aligned_dem_fn is None
        assert icesat.aligned_dem_fn is None
        assert not fake_aligned.exists(), "aligned DEM should be cleaned up"

    def test_align_and_evaluate_success(self, icesat, monkeypatch, tmp_path):
        """Substantial p50 reduction maps to success and retains the aligned DEM."""
        fake_aligned = tmp_path / "fake-DEM_pc_align_translated.tif"
        fake_aligned.write_bytes(b"dummy")

        def fake_alignment_report(self, **kwargs):
            self.aligned_dem_fn = str(fake_aligned)
            self.alignment_report_df = pd.DataFrame(
                [
                    {
                        "key": "all",
                        "p16_beg": 1.0,
                        "p50_beg": 2.0,
                        "p84_beg": 3.0,
                        "p16_end": 0.4,
                        "p50_end": 0.5,
                        "p84_end": 1.0,
                        "north_shift": 0.1,
                        "east_shift": 0.2,
                        "down_shift": -1.5,
                        "translation_magnitude": 1.5,
                    }
                ]
            )

        def fake_atl06sr_to_dem_dh(self, n_sigma=3):
            return

        monkeypatch.setattr(Altimetry, "alignment_report", fake_alignment_report)
        monkeypatch.setattr(Altimetry, "atl06sr_to_dem_dh", fake_atl06sr_to_dem_dh)
        result = icesat.align_and_evaluate(improvement_threshold_pct=5.0)
        assert result.status == "success"
        assert result.aligned_dem_fn == str(fake_aligned)
        assert fake_aligned.exists(), "aligned DEM should be retained"
        assert result.improvement_pct is not None
        assert result.improvement_pct > 5.0


class TestLazySlideruleInit:
    """Test that SlideRule is not initialized during Altimetry construction."""

    def test_no_sliderule_on_init(self):
        alt = Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )
        assert alt._sliderule_initialized is False

    def test_planetary_points_initialized(self):
        alt = Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )
        assert alt.planetary_points is None


class TestResolveTimeRange:
    @pytest.fixture
    def alt(self):
        return Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

    def test_default_is_all(self, alt):
        t0, t1, resolved = alt._resolve_time_range()
        assert resolved is None
        assert t0 == ICESAT2_MISSION_START.strftime("%Y-%m-%dT%H:%M:%SZ")
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        assert t1 == today.strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_explicit_date(self, alt):
        t0, t1, resolved = alt._resolve_time_range(
            time_range="buffered", scene_date="2022-06-15"
        )
        assert resolved == datetime(2022, 6, 15, tzinfo=timezone.utc)
        assert t0 == "2021-06-15T00:00:00Z"
        assert t1 == "2023-06-15T00:00:00Z"

    def test_t0_clamped_to_mission_start(self, alt):
        t0, t1, resolved = alt._resolve_time_range(
            time_range="buffered", scene_date="2019-03-01"
        )
        assert resolved == datetime(2019, 3, 1, tzinfo=timezone.utc)
        assert t0 == ICESAT2_MISSION_START.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert t1 == "2020-02-29T00:00:00Z"

    def test_pre_mission_date_triggers_fallback(self, alt):
        t0, t1, resolved = alt._resolve_time_range(
            time_range="buffered", scene_date="2015-01-01"
        )
        # Falls back to "all"
        assert resolved is None
        assert t0 == ICESAT2_MISSION_START.strftime("%Y-%m-%dT%H:%M:%SZ")
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        assert t1 == today.strftime("%Y-%m-%dT%H:%M:%SZ")

    def test_auto_detect_from_xml(self, alt):
        t0, t1, resolved = alt._resolve_time_range(time_range="buffered")
        # Test data XMLs have dates around 2022-04-19
        assert resolved is not None
        assert resolved.year == 2022
        assert resolved.month == 4

    def test_custom_buffer(self, alt):
        t0, t1, resolved = alt._resolve_time_range(
            time_range="buffered", scene_date="2022-06-15", time_buffer_days=30
        )
        assert t0 == "2022-05-16T00:00:00Z"
        assert t1 == "2022-07-15T00:00:00Z"

    def test_explicit_t0_t1(self, alt):
        t0, t1, resolved = alt._resolve_time_range(
            time_range="buffered", t0="2021-01-01", t1="2023-01-01"
        )
        assert resolved is None
        assert t0 == "2021-01-01T00:00:00Z"
        assert t1 == "2023-01-01T00:00:00Z"

    def test_cached_attributes(self, alt):
        alt._resolve_time_range(time_range="buffered", scene_date="2022-06-15")
        assert alt._scene_date == datetime(2022, 6, 15, tzinfo=timezone.utc)
        assert alt._t0 == datetime(2021, 6, 15, tzinfo=timezone.utc)
        assert alt._t1 == datetime(2023, 6, 15, tzinfo=timezone.utc)


class TestPlanetaryDh:
    """Test planetary altimetry dh computation with the Earth test DEM."""

    @pytest.fixture
    def alt_with_points(self):
        """Create Altimetry instance with mock planetary points."""
        alt = Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

        # Get the DEM bounds in latlon to create points within the DEM extent
        from asp_plot.utils import Raster

        raster = Raster(alt.dem_fn)
        bounds = raster.get_bounds(latlon=True, json_format=False)
        # bounds = (min_lon, min_lat, max_lon, max_lat)
        lon_min, lat_min, lon_max, lat_max = bounds

        # Create a small grid of points inside the DEM
        lons = np.linspace(lon_min + 0.001, lon_max - 0.001, 5)
        lats = np.linspace(lat_min + 0.001, lat_max - 0.001, 5)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lon_flat = lon_grid.flatten()
        lat_flat = lat_grid.flatten()

        # Assign mock height values (just use 100m for simplicity)
        heights = np.full(len(lon_flat), 100.0)

        gdf = gpd.GeoDataFrame(
            {"height": heights, "lon": lon_flat, "lat": lat_flat},
            geometry=[Point(x, y) for x, y in zip(lon_flat, lat_flat)],
            crs="EPSG:4326",
        )
        alt.planetary_points = gdf
        return alt

    def test_planetary_to_dem_dh(self, alt_with_points):
        """Test that dh is computed and stored."""
        alt_with_points.planetary_to_dem_dh()
        assert "dem_height" in alt_with_points.planetary_points.columns
        assert "altimetry_minus_dem" in alt_with_points.planetary_points.columns
        # At least some valid dh values
        valid = alt_with_points.planetary_points["altimetry_minus_dem"].dropna()
        assert len(valid) > 0

    def test_mapview_plot_planetary_to_dem(self, alt_with_points):
        """Test that map view plot runs without error."""
        alt_with_points.planetary_to_dem_dh()
        try:
            alt_with_points.mapview_plot_planetary_to_dem()
        except Exception as e:
            pytest.fail(f"mapview_plot_planetary_to_dem raised: {e}")

    def test_histogram_planetary_to_dem(self, alt_with_points):
        """Test that histogram plot runs without error."""
        alt_with_points.planetary_to_dem_dh()
        try:
            alt_with_points.histogram_planetary_to_dem()
        except Exception as e:
            pytest.fail(f"histogram_planetary_to_dem raised: {e}")

    def test_to_csv_for_pc_align_planetary(self, alt_with_points, tmp_path):
        """to_csv_for_pc_align_planetary writes lon/lat/radius_m only."""
        # Add the radius_m column the loaders normally populate
        alt_with_points.planetary_points["radius_m"] = (
            alt_with_points.planetary_points["height"] + 6_378_137.0
        )
        alt_with_points.directory = str(tmp_path)
        csv_fn = alt_with_points.to_csv_for_pc_align_planetary()
        assert os.path.exists(csv_fn)
        df = pd.read_csv(csv_fn)
        assert list(df.columns) == ["lon", "lat", "radius_m"]
        assert len(df) > 0
        # No NaNs in the exported file
        assert not df.isna().any().any()

    def test_to_csv_for_pc_align_planetary_requires_radius(
        self, alt_with_points, tmp_path
    ):
        """Missing radius_m column raises a clear error."""
        alt_with_points.directory = str(tmp_path)
        # Strip radius_m if present
        if "radius_m" in alt_with_points.planetary_points.columns:
            alt_with_points.planetary_points = alt_with_points.planetary_points.drop(
                columns=["radius_m"]
            )
        with pytest.raises(ValueError, match="radius_m"):
            alt_with_points.to_csv_for_pc_align_planetary()

    def test_align_and_evaluate_planetary_no_points(self, alt_with_points):
        """No planetary points loaded → insufficient_points status."""
        alt_with_points.planetary_points = None
        result = alt_with_points.align_and_evaluate_planetary()
        assert result.status == "insufficient_points"
        assert result.aligned_dem_fn is None

    def test_align_and_evaluate_planetary_too_few_overlap(
        self, alt_with_points, tmp_path
    ):
        """Few overlapping valid dh points → insufficient_points status."""
        alt_with_points.directory = str(tmp_path)
        # All points are nominally valid in the fixture; force minimum_points
        # higher than the available count to take the early-exit branch.
        alt_with_points.planetary_points["radius_m"] = (
            alt_with_points.planetary_points["height"] + 6_378_137.0
        )
        result = alt_with_points.align_and_evaluate_planetary(minimum_points=10_000)
        assert result.status == "insufficient_points"
        assert result.aligned_dem_fn is None


class TestLoadPlanetaryCsv:
    """Test LOLA and MOLA CSV loading and validation."""

    @pytest.fixture
    def alt(self):
        return Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

    def test_load_lola_csv(self, alt, tmp_path):
        """Test LOLA CSV parsing with topo CSV (no Pt_Radius)."""
        csv = tmp_path / "lola.csv"
        csv.write_text(
            "Pt_Longitude, Pt_Latitude, Topography\n"
            " 15.3287,  -9.6003,     295.81\n"
            " 15.3286,  -9.6021,     297.16\n"
            " 15.3286,  -9.6039,     300.09\n"
        )
        alt._load_lola_csv(str(csv))
        assert alt.planetary_points is not None
        assert len(alt.planetary_points) == 3
        assert "height" in alt.planetary_points.columns
        assert "radius_m" in alt.planetary_points.columns
        # Height should equal topography directly for LOLA
        assert alt.planetary_points["height"].iloc[0] == pytest.approx(295.81)
        # radius_m back-computed against the IAU 1737.4 km sphere
        assert alt.planetary_points["radius_m"].iloc[0] == pytest.approx(1737695.81)

    def test_load_lola_csv_pts_radius_km(self, alt, tmp_path):
        """LOLA Point per Row CSV has Pt_Radius in km — auto-convert to m."""
        csv = tmp_path / "lola_pts.csv"
        csv.write_text(
            "Pt_Longitude, Pt_Latitude, Pt_Radius\n"
            " 15.3287,  -9.6003, 1737.668720\n"
            " 15.3286,  -9.6021, 1737.666748\n"
        )
        alt._load_lola_csv(str(csv))
        # Pt_Radius is preferred; km auto-detected and converted to m.
        assert alt.planetary_points["radius_m"].iloc[0] == pytest.approx(
            1737668.720, abs=1e-3
        )
        assert alt.planetary_points["height"].iloc[0] == pytest.approx(
            268.720, abs=1e-3
        )

    def test_load_mola_csv(self, alt, tmp_path):
        """Test MOLA CSV parsing with PLANET_RAD column (the supported path)."""
        csv = tmp_path / "mola.csv"
        csv.write_text(
            "LONG_EAST,LAT_NORTH, TOPOGRAPHY, PLANET_RAD,            UTC\n"
            "137.13264, -4.91750,   -4499.73, 3391690.27,1999-08-31T19:13:24.847\n"
            "137.13197, -4.91240,   -4505.07, 3391684.93,1999-08-31T19:13:24.947\n"
        )
        alt._load_mola_csv(str(csv))
        assert alt.planetary_points is not None
        assert len(alt.planetary_points) == 2
        assert "height" in alt.planetary_points.columns
        assert "radius_m" in alt.planetary_points.columns
        # height = PLANET_RAD - 3,396,190 (IAU sphere)
        assert alt.planetary_points["height"].iloc[0] == pytest.approx(-4499.73)
        assert alt.planetary_points["radius_m"].iloc[0] == pytest.approx(3391690.27)

    def test_load_mola_topo_only_raises(self, alt, tmp_path):
        """Test that a *_topo_csv.csv (no PLANET_RAD) is rejected with help."""
        csv = tmp_path / "mola_topo.csv"
        csv.write_text(
            "LONG_EAST,LAT_NORTH, TOPOGRAPHY,            UTC\n"
            "137.13264, -4.91750,   -4499.73,1999-08-31T19:13:24.847\n"
        )
        with pytest.raises(ValueError, match="PLANET_RAD"):
            alt._load_mola_csv(str(csv))

    def test_load_mola_csv_lon_conversion(self, alt, tmp_path):
        """Test that MOLA 0-360 longitude is converted to -180/180."""
        csv = tmp_path / "mola.csv"
        csv.write_text(
            "LONG_EAST,LAT_NORTH, TOPOGRAPHY, PLANET_RAD,            UTC\n"
            "270.0, 10.0, -3000.0, 3393190.0, 1999-01-01T00:00:00\n"
        )
        alt._load_mola_csv(str(csv))
        assert alt.planetary_points["lon"].iloc[0] == pytest.approx(-90.0)

    def test_load_csv_empty_raises(self, alt, tmp_path):
        """Test that empty CSV raises ValueError."""
        csv = tmp_path / "empty.csv"
        csv.write_text("Pt_Longitude, Pt_Latitude, Topography\n")
        with pytest.raises(ValueError, match="empty"):
            alt._load_lola_csv(str(csv))

    def test_load_csv_wrong_columns_raises(self, alt, tmp_path):
        """Test that wrong columns raise ValueError with helpful message."""
        csv = tmp_path / "wrong.csv"
        csv.write_text("col_a, col_b, col_c\n1,2,3\n")
        with pytest.raises(ValueError, match="planetary radius"):
            alt._load_lola_csv(str(csv))

    def test_load_planetary_csv_earth_raises(self, alt, tmp_path):
        """Test that load_planetary_csv rejects Earth DEMs."""
        csv = tmp_path / "dummy.csv"
        csv.write_text("a,b,c\n1,2,3\n")
        with pytest.raises(ValueError, match="ICESat-2"):
            alt.load_planetary_csv(str(csv))
