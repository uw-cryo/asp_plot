import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest

from asp_plot.bundle_adjust import (
    PlotBundleAdjustCameras,
    PlotBundleAdjustFiles,
    ReadBundleAdjustCameras,
    ReadBundleAdjustFiles,
    _normalize_camera_id,
)

matplotlib.use("Agg")


class TestBundleAdjust:
    @pytest.fixture
    def ba_files(self):
        directory = "tests/test_data"
        ba_directory = "ba"
        return ReadBundleAdjustFiles(directory, ba_directory)

    @pytest.fixture
    def ba_files_no_mapproj_dem(self):
        directory = "tests/test_data"
        ba_directory = "ba_no_mapproj_dem"
        return ReadBundleAdjustFiles(directory, ba_directory)

    def test_get_initial_final_residuals_gdfs(self, ba_files):
        resid_initial, resid_final = ba_files.get_initial_final_residuals_gdfs()
        assert isinstance(resid_initial, gpd.GeoDataFrame)
        assert isinstance(resid_final, gpd.GeoDataFrame)

    def test_get_initial_final_geodiff_gdfs(self, ba_files):
        geodiff_initial, geodiff_final = ba_files.get_initial_final_residuals_gdfs(
            residuals_in_meters=True
        )
        assert isinstance(geodiff_initial, gpd.GeoDataFrame)
        assert isinstance(geodiff_final, gpd.GeoDataFrame)

    def test_get_mapproj_residuals_gdf(self, ba_files):
        resid_mapprojected_gdf = ba_files.get_mapproj_residuals_gdf()
        assert isinstance(resid_mapprojected_gdf, gpd.GeoDataFrame)

    def test_get_propagated_triangulation_uncert_df(self, ba_files):
        resid_triangulation_uncert_df = (
            ba_files.get_propagated_triangulation_uncert_df()
        )
        assert isinstance(resid_triangulation_uncert_df, pd.DataFrame)

    def test_plot_n_gdfs(self, ba_files):
        resid_initial, resid_final = ba_files.get_initial_final_residuals_gdfs()
        try:
            PlotBundleAdjustFiles([resid_initial, resid_final]).plot_n_gdfs(
                column_name="mean_residual"
            )
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_get_initial_final_geodiff_gdfs_no_mapproj_dem(
        self, ba_files_no_mapproj_dem
    ):
        """Test that geodiff gracefully fails when no --mapproj-dem was used in bundle_adjust."""
        with pytest.raises(ValueError) as exc_info:
            ba_files_no_mapproj_dem.get_initial_final_geodiff_gdfs()
        assert "could not be generated" in str(exc_info.value)

    def test_residuals_still_work_without_mapproj_dem(self, ba_files_no_mapproj_dem):
        """Test that residual GeoDataFrames can still be read even without --mapproj-dem."""
        resid_initial, resid_final = (
            ba_files_no_mapproj_dem.get_initial_final_residuals_gdfs()
        )
        assert isinstance(resid_initial, gpd.GeoDataFrame)
        assert isinstance(resid_final, gpd.GeoDataFrame)


class TestBundleAdjustCameras:
    @pytest.fixture
    def cam_reader(self):
        # ba_cams holds a stereo pair of CSM cameras with .adjust,
        # .adjusted_state.json, and a camera_offsets.txt fixture.
        return ReadBundleAdjustCameras("tests/test_data", "ba_cams")

    def test_normalize_camera_id(self):
        assert (
            _normalize_camera_id("run-1040010074793300_corr.tif") == "1040010074793300"
        )
        assert (
            _normalize_camera_id("run-out-Band3B.adjusted_state.json") == "out-Band3B"
        )
        assert (
            _normalize_camera_id("1040010075633C00.adjusted_state.adjust")
            == "1040010075633C00"
        )

    def test_read_adjust_file(self, cam_reader):
        translation, rotation = cam_reader.read_adjust_file(
            "tests/test_data/ba_cams/1040010074793300.adjust"
        )
        assert translation.shape == (3,)
        # Nearly-identity adjustment: rotation magnitude should be tiny.
        assert rotation.magnitude() < 0.01

    def test_get_camera_offsets_df(self, cam_reader):
        df = cam_reader.get_camera_offsets_df()
        assert isinstance(df, pd.DataFrame)
        assert {"horizontal_offset_m", "vertical_offset_m", "camera_id"} <= set(
            df.columns
        )

    def test_get_camera_optimization_gdf(self, cam_reader):
        gdf = cam_reader.get_camera_optimization_gdf(map_crs=32619)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        for col in [
            "camera_id",
            "t_east",
            "t_north",
            "t_up",
            "t_horizontal",
            "adj_roll",
            "adj_pitch",
            "adj_yaw",
            "horizontal_offset_m",
            "vertical_offset_m",
            "offsets_from_asp",
        ]:
            assert col in gdf.columns
        # camera_offsets.txt fixture is present, so magnitudes come from ASP.
        assert gdf.offsets_from_asp.all()
        assert gdf.crs.to_epsg() == 32619

    def test_optimization_gdf_fallback_without_offsets(self, cam_reader, monkeypatch):
        """Without camera_offsets.txt, magnitudes fall back to the .adjust translation."""
        monkeypatch.setattr(cam_reader, "get_camera_offsets_df", lambda: None)
        gdf = cam_reader.get_camera_optimization_gdf(map_crs=32619)
        assert not gdf.offsets_from_asp.any()
        # Fallback horizontal offset equals the translation horizontal magnitude.
        assert np.allclose(gdf.horizontal_offset_m, gdf.t_horizontal)

    def test_get_camera_optimization_gdf_raises_without_state(self):
        # The plain "ba" residual dir has no .adjusted_state.json files.
        reader = ReadBundleAdjustCameras("tests/test_data", "ba")
        with pytest.raises(ValueError):
            reader.get_camera_optimization_gdf()

    def test_plot_methods(self, cam_reader):
        gdf = cam_reader.get_camera_optimization_gdf(map_crs=32619)
        plotter = PlotBundleAdjustCameras(gdf, title="Test cameras")
        try:
            plotter.plot_position_change_quiver()
            plotter.plot_center_offset_bars()
            plotter.plot_orientation_change_quiver()
            plotter.summary_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
