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
    _camera_label,
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


class TestStemNarrowing:
    """An ASP-style prefix stem narrows the globs to that run's outputs (#60)."""

    def test_matching_stem_finds_files(self):
        ba_files = ReadBundleAdjustFiles("tests/test_data", "ba", stem="ba")
        initial, final = ba_files.get_csv_paths()
        assert initial.endswith("ba-initial_residuals_pointmap.csv")
        assert final.endswith("ba-final_residuals_pointmap.csv")

    def test_non_matching_stem_finds_nothing(self):
        ba_files = ReadBundleAdjustFiles("tests/test_data", "ba", stem="other_run")
        with pytest.raises(ValueError, match="not found"):
            ba_files.get_csv_paths()


class TestBundleAdjustCameras:
    @pytest.fixture
    def cam_reader(self):
        # ba_cams holds a stereo pair of CSM cameras with .adjust,
        # .adjusted_state.json, and a camera_offsets.txt fixture.
        return ReadBundleAdjustCameras("tests/test_data", "ba_cams")

    def test_camera_label(self):
        # Only deterministic camera extensions are stripped; the ASP output
        # prefix (run-, ba_mvs_csm-, ...) is left intact -- no brittle guessing.
        assert _camera_label("run-out-Band3B.adjusted_state.json") == "run-out-Band3B"
        assert (
            _camera_label("ba_mvs_csm-10300100D044F700.r100.adjusted_state.json")
            == "ba_mvs_csm-10300100D044F700.r100"
        )
        assert _camera_label("10300100D044F700.r100.xml") == "10300100D044F700.r100"

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
        assert {"image", "horizontal_offset_m", "vertical_offset_m"} <= set(df.columns)

    def test_offsets_associated_by_camera_list_order(self, cam_reader):
        # Positional association via camera_list.txt: no filename munging.
        mapping = cam_reader._offsets_by_camera_basename()
        assert mapping["1040010074793300.adjusted_state.json"][0] == pytest.approx(
            0.72481032
        )
        assert mapping["1040010075633C00.adjusted_state.json"][1] == pytest.approx(
            0.20482915
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
        """Without ASP offsets, magnitudes fall back to the .adjust translation."""
        monkeypatch.setattr(cam_reader, "_offsets_by_camera_basename", lambda: None)
        gdf = cam_reader.get_camera_optimization_gdf(map_crs=32619)
        assert not gdf.offsets_from_asp.any()
        # Fallback horizontal offset equals the translation horizontal magnitude.
        assert np.allclose(gdf.horizontal_offset_m, gdf.t_horizontal)

    def test_malformed_adjust_is_skipped(self, tmp_path):
        """A corrupt/truncated .adjust skips that camera instead of crashing."""
        import shutil
        from glob import glob

        ba = tmp_path / "ba"
        ba.mkdir()
        src = "tests/test_data/ba_cams"
        for f in glob(f"{src}/*.adjusted_state.json") + glob(f"{src}/*.adjust"):
            shutil.copy(f, ba)
        # Truncate one .adjust to drop the quaternion line.
        victim = sorted(glob(f"{ba}/*.adjust"))[0]
        with open(victim, "w") as fh:
            fh.write("0.1 0.2 0.3\n")

        reader = ReadBundleAdjustCameras(str(tmp_path), "ba")
        gdf = reader.get_camera_optimization_gdf()
        # One camera dropped, the other survives (no exception raised).
        assert len(gdf) == 1

    def test_digitalglobe_case_reads_center_from_xml(self, tmp_path):
        """DG runs have no *.adjusted_state.json; the center comes from the .xml."""
        import shutil
        from glob import glob

        ba = tmp_path / "ba"
        ba.mkdir()
        src = "tests/test_data/ba_cams"
        # Copy only the .adjust deltas and the original .xml cameras (no state json),
        # mimicking a DigitalGlobe bundle_adjust output.
        for f in glob(f"{src}/*.adjust") + glob(f"{src}/*.xml"):
            shutil.copy(f, ba)

        reader = ReadBundleAdjustCameras(str(tmp_path), "ba")
        gdf = reader.get_camera_optimization_gdf(map_crs=32619)
        assert len(gdf) == 2
        assert not gdf.offsets_from_asp.any()  # no camera_offsets.txt here
        # Centers must be real (finite) ECEF-derived points, not NaN.
        assert gdf.geometry.x.notna().all() and gdf.geometry.y.notna().all()

    def test_digitalglobe_missing_xml_skips(self, tmp_path):
        """A DG .adjust with no locatable original camera is skipped, not fatal."""
        import shutil
        from glob import glob

        ba = tmp_path / "ba"
        ba.mkdir()
        src = "tests/test_data/ba_cams"
        adjusts = sorted(glob(f"{src}/*.adjust"))
        xmls = sorted(glob(f"{src}/*.xml"))
        for f in adjusts:  # both deltas
            shutil.copy(f, ba)
        shutil.copy(xmls[0], ba)  # only one of two original cameras

        reader = ReadBundleAdjustCameras(str(tmp_path), "ba")
        gdf = reader.get_camera_optimization_gdf()
        assert len(gdf) == 1  # the camera without an .xml is skipped

    def test_get_camera_optimization_gdf_raises_without_adjust(self):
        # The plain "ba" residual dir has no .adjust files.
        reader = ReadBundleAdjustCameras("tests/test_data", "ba")
        with pytest.raises(ValueError):
            reader.get_camera_optimization_gdf()

    def test_plot_methods(self, cam_reader):
        gdf = cam_reader.get_camera_optimization_gdf(map_crs=32619)
        plotter = PlotBundleAdjustCameras(gdf, title="Test cameras")
        try:
            plotter.plot_center_offset_bars()
            plotter.plot_orientation_bars()
            plotter.summary_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_identity_run_draws_note(self, cam_reader):
        """All-zero changes still draw the panels, with a 'no camera change' note."""
        gdf = cam_reader.get_camera_optimization_gdf(map_crs=32619)
        for col in [
            "horizontal_offset_m",
            "vertical_offset_m",
            "adj_roll",
            "adj_pitch",
            "adj_yaw",
        ]:
            gdf[col] = 0.0
        gdf["adj_roll"] = -0.0  # a signed zero, as real identity .adjust files give
        plotter = PlotBundleAdjustCameras(gdf)
        assert plotter.is_identity
        fig = plotter.summary_plot()
        notes = [
            t.get_text()
            for ax in fig.axes
            for t in ax.texts
            if "no camera change" in t.get_text()
        ]
        assert len(notes) == 2  # one per bar panel
        assert all("identity" in n for n in notes)

    def test_fmt_deg_signed_zero(self):
        from asp_plot.bundle_adjust import _fmt_deg

        assert _fmt_deg(-0.0) == "+0°"
        assert _fmt_deg(0.00039) == "+0.00039°"
        assert _fmt_deg(-0.0161) == "-0.016°"
