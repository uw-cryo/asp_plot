import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from affine import Affine
from shapely.geometry import Point

from asp_plot.altimetry import Altimetry
from asp_plot.selections import (
    FigureSelections,
    bbox_to_pixel_offset,
    pixel_window_to_bbox,
    read_selections_yaml,
    reproject_bbox,
    write_selections_yaml,
)
from asp_plot.stereo import StereoPlotter

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# FigureSelections YAML round-trip
# ---------------------------------------------------------------------------


class TestFigureSelectionsRoundTrip:
    def test_round_trip(self, tmp_path):
        selections = FigureSelections(
            asp_plot_version="9.9.9",
            dem_filename="stereo/run-DEM.tif",
            map_crs="EPSG:32610",
            detailed_hillshade={
                "subset_km": 1.0,
                "intersection_error_percentiles": [16, 50, 84],
                "dem_crs": "EPSG:32610",
                "clips": [
                    {
                        "label": "low",
                        "bbox": [0.0, 0.0, 10.0, 10.0],
                        "pixel_offset": [0, 0],
                    },
                ],
            },
            icesat2={
                "request": {"res": 20, "len": 40},
                "parquet_cache": {"all": "stereo/atl06sr_all.parquet"},
                "profile_track": {"rgt": 12, "cycle": 5, "spot": 3},
                "segments": {
                    "best": {"start_km": 0.0, "end_km": 1.0},
                    "worst": {"start_km": 4.0, "end_km": 5.0},
                },
            },
        )
        path = tmp_path / "report_figure_selections.yml"
        write_selections_yaml(str(path), selections)
        assert path.exists()

        loaded = read_selections_yaml(str(path))
        assert loaded.asp_plot_version == "9.9.9"
        assert loaded.dem_filename == "stereo/run-DEM.tif"
        assert loaded.map_crs == "EPSG:32610"
        assert loaded.detailed_hillshade["subset_km"] == 1.0
        assert loaded.detailed_hillshade["clips"][0]["label"] == "low"
        assert loaded.icesat2["profile_track"] == {"rgt": 12, "cycle": 5, "spot": 3}
        assert loaded.icesat2["segments"]["worst"]["end_km"] == 5.0

    def test_from_dict_empty(self):
        sel = FigureSelections.from_dict(None)
        assert sel.detailed_hillshade is None
        assert sel.icesat2 is None


# ---------------------------------------------------------------------------
# Geometry helpers: clip bbox <-> pixel window
# ---------------------------------------------------------------------------


class TestClipGeometry:
    def test_bbox_pixel_round_trip(self):
        # North-up transform: 2 m GSD, origin at (500000, 4000000)
        gsd = 2.0
        transform = Affine(gsd, 0, 500000.0, 0, -gsd, 4000000.0)
        row, col, n = 30, 50, 100

        bbox = pixel_window_to_bbox(transform, row, col, n, n)
        # bbox is [xmin, ymin, xmax, ymax]
        assert bbox[0] < bbox[2] and bbox[1] < bbox[3]

        back_row, back_col = bbox_to_pixel_offset(transform, bbox)
        assert (back_row, back_col) == (row, col)

    def test_bbox_to_pixel_offset_clamps_negative(self):
        gsd = 2.0
        transform = Affine(gsd, 0, 500000.0, 0, -gsd, 4000000.0)
        # A bbox above/left of the raster origin -> negative indices clamped to 0
        bbox = [499000.0, 4001000.0, 499500.0, 4001500.0]
        row, col = bbox_to_pixel_offset(transform, bbox)
        assert row >= 0 and col >= 0

    def test_reproject_bbox_noop_same_or_missing_crs(self):
        bbox = [500000.0, 4000000.0, 500500.0, 4000500.0]
        assert reproject_bbox(bbox, None, "EPSG:32610") == bbox
        assert reproject_bbox(bbox, "EPSG:32610", None) == bbox
        assert reproject_bbox(bbox, "EPSG:32610", "EPSG:32610") == bbox

    def test_reproject_bbox_round_trip(self):
        # UTM 10N -> WGS84 -> back should recover the original bbox closely
        bbox = [500000.0, 4000000.0, 501000.0, 4001000.0]
        wgs = reproject_bbox(bbox, "EPSG:32610", "EPSG:4326")
        # Reprojected coords are degrees near (-123, 36)
        assert -124 < wgs[0] < -122 and 35 < wgs[1] < 37
        back = reproject_bbox(wgs, "EPSG:4326", "EPSG:32610")
        np.testing.assert_allclose(back, bbox, atol=1.0)


# ---------------------------------------------------------------------------
# Detailed-hillshade clip replay (stereo)
# ---------------------------------------------------------------------------


class TestHillshadeClipReplay:
    @pytest.fixture
    def stereo_plotter(self):
        return StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            dem_gsd=1,
            reference_dem="tests/test_data/ref_dem.tif",
            title="Stereo Results",
        )

    def test_records_clips(self, stereo_plotter):
        stereo_plotter.plot_detailed_hillshade(subset_km=10)
        clips = stereo_plotter.detailed_hillshade_clips
        assert len(clips) == 3
        for clip in clips:
            assert "bbox" in clip and len(clip["bbox"]) == 4
            assert "label" in clip and "pixel_offset" in clip

    def test_replay_clips_match(self, stereo_plotter):
        # First run records clips; second run replays them and should produce
        # the same boxes (issue #121 reproducibility).
        stereo_plotter.plot_detailed_hillshade(subset_km=10)
        first = [c["bbox"] for c in stereo_plotter.detailed_hillshade_clips]

        stereo_plotter.plot_detailed_hillshade(subset_km=10, clip_windows=first)
        second = [c["bbox"] for c in stereo_plotter.detailed_hillshade_clips]

        assert len(second) == 3
        for b1, b2 in zip(first, second):
            np.testing.assert_allclose(b1, b2, rtol=0, atol=1.0)

    def test_out_of_bounds_clip_falls_back(self, stereo_plotter):
        # A bbox far outside the DEM should warn and fall back to auto, still
        # yielding three clips.
        bogus = [[1e9, 1e9, 1e9 + 10, 1e9 + 10]] * 3
        stereo_plotter.plot_detailed_hillshade(subset_km=10, clip_windows=bogus)
        assert len(stereo_plotter.detailed_hillshade_clips) == 3


# ---------------------------------------------------------------------------
# Segment pinning (_find_best_worst_segments override)
# ---------------------------------------------------------------------------


def _synthetic_track():
    x_atc = np.arange(0, 2000, 20, dtype=float)  # 100 points spanning 1980 m
    n = len(x_atc)
    rng = np.linspace(0, 1, n)
    data = {
        "x_atc": x_atc,
        "dem_height": 100.0 + rng,
        "icesat_minus_dem": np.sin(rng * 10.0),
        "h_mean": 100.0 + rng,
        "geometry": [Point(-120.0 + i * 1e-4, 39.0 + i * 1e-4) for i in range(n)],
    }
    # No CRS set: the segment-override path is CRS-agnostic, and avoiding a CRS
    # keeps the test independent of the local PROJ database.
    return gpd.GeoDataFrame(data, geometry="geometry")


class TestSegmentOverride:
    @pytest.fixture
    def icesat(self):
        return Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

    def test_segment_override_applied(self, icesat):
        track = _synthetic_track()
        override = {
            "best": {"start_km": 0.0, "end_km": 1.0},
            "worst": {"start_km": 1.0, "end_km": 1.9},
        }
        seg = icesat._find_best_worst_segments(track, segment_override=override)
        assert seg is not None
        assert seg["seg_best_start_km"] == pytest.approx(0.0, abs=1e-6)
        assert seg["seg_best_end_km"] == pytest.approx(1.0, abs=1e-6)
        assert seg["seg_worst_start_km"] == pytest.approx(1.0, abs=1e-6)
        assert seg["seg_worst_end_km"] == pytest.approx(1.9, abs=1e-6)
        # Masks select the expected along-track ranges
        assert seg["seg_best_mask"].sum() > 0
        assert seg["seg_worst_mask"].sum() > 0

    def test_segment_override_bad_falls_back(self, icesat):
        track = _synthetic_track()
        # Missing "worst" -> should fall back to automatic scoring, not raise
        seg = icesat._find_best_worst_segments(
            track, segment_override={"best": {"start_km": 0.0, "end_km": 1.0}}
        )
        assert seg is not None
        assert "seg_best_start_km" in seg


# ---------------------------------------------------------------------------
# Parquet reuse + selection bundling
# ---------------------------------------------------------------------------


class TestParquetReuseAndSelections:
    @pytest.fixture
    def icesat(self):
        return Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

    def test_load_from_parquet(self, icesat):
        paths = {"all": "tests/test_data/icesat_data/atl06sr_all.parquet"}
        loaded = icesat.load_atl06sr_from_parquet(paths)
        assert loaded is True
        assert "all" in icesat.atl06sr_processing_levels
        assert len(icesat.atl06sr_processing_levels["all"]) > 0
        assert icesat.atl06sr_parquet_paths["all"].endswith("atl06sr_all.parquet")

    def test_load_from_parquet_missing(self, icesat):
        loaded = icesat.load_atl06sr_from_parquet({"all": "does/not/exist.parquet"})
        assert loaded is False

    def test_get_altimetry_selections_bundles_request(self, icesat, monkeypatch):
        # Set request + parquet metadata, stub track resolution so we test the
        # bundling without the pyproj-dependent dh pipeline.
        icesat.atl06sr_request_parms = {"res": 20, "len": 40, "time_range": "all"}
        icesat.atl06sr_parquet_paths = {"all": "stereo/atl06sr_all.parquet"}
        monkeypatch.setattr(icesat, "_resolve_best_track", lambda key="all": None)

        sel = icesat.get_altimetry_selections("all")
        assert sel["request"]["res"] == 20
        assert sel["parquet_cache"]["all"].endswith("atl06sr_all.parquet")
        assert "profile_track" not in sel
