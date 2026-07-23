import os

import matplotlib
import pytest

from asp_plot.stereo_geometry import StereoGeometryPlotter, camera_files_from_stereo_run
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser

matplotlib.use("Agg")


class TestCameraFilesFromStereoRun:
    LOG_TEMPLATE = (
        "ASP 3.8.0-alpha\n"
        "Build ID: 39e4f4a\n"
        "Build date: 2026-06-26\n"
        "\n"
        "{command}\n"
        "\n"
        "uname -a\n"
        "Darwin test\n"
    )

    def _make_run(self, tmp_path, command, cameras=("left.xml", "right.xml")):
        for camera in cameras:
            (tmp_path / camera).write_text("<xml/>")
        stereo_dir = tmp_path / "stereo"
        stereo_dir.mkdir()
        (stereo_dir / "run-log-stereo_tri-01-01-0000-1.txt").write_text(
            self.LOG_TEMPLATE.format(command=command)
        )
        return tmp_path

    def test_resolves_command_cameras(self, tmp_path):
        directory = self._make_run(
            tmp_path,
            "stereo_tri --threads 8 left.tif right.tif "
            "left.xml right.xml stereo/run",
        )
        files = camera_files_from_stereo_run(str(directory), "stereo")
        assert [os.path.basename(f) for f in files] == ["left.xml", "right.xml"]
        assert all(os.path.isfile(f) for f in files)

    def test_basename_fallback_for_stale_paths(self, tmp_path):
        # Command recorded with paths that no longer exist; the basenames
        # are still present in the processing directory.
        directory = self._make_run(
            tmp_path,
            "stereo_tri left.tif right.tif "
            "/old/path/left.xml /old/path/right.xml stereo/run",
        )
        files = camera_files_from_stereo_run(str(directory), "stereo")
        assert [os.path.basename(f) for f in files] == ["left.xml", "right.xml"]

    def test_missing_camera_returns_none(self, tmp_path):
        directory = self._make_run(
            tmp_path,
            "stereo_tri left.tif right.tif left.xml gone.xml stereo/run",
        )
        assert camera_files_from_stereo_run(str(directory), "stereo") is None

    def test_non_xml_cameras_return_none(self, tmp_path):
        # CSM runs pass .json camera states; there is nothing to scope to.
        directory = self._make_run(
            tmp_path,
            "stereo_tri left.tif right.tif "
            "ba/left.adjusted_state.json ba/right.adjusted_state.json stereo/run",
        )
        assert camera_files_from_stereo_run(str(directory), "stereo") is None

    def test_no_logs_returns_none(self, tmp_path):
        (tmp_path / "stereo").mkdir()
        assert camera_files_from_stereo_run(str(tmp_path), "stereo") is None
        assert camera_files_from_stereo_run(None, "stereo") is None
        assert camera_files_from_stereo_run(str(tmp_path), None) is None

    def test_newest_log_wins(self, tmp_path):
        # A rerun into the same stereo directory must scope to the current
        # run's cameras: the newest log wins, not the alphabetically-first
        # (ASP log names embed no year to sort on).
        directory = self._make_run(
            tmp_path,
            "stereo_tri old_l.tif old_r.tif old_left.xml old_right.xml stereo/run",
            cameras=("old_left.xml", "old_right.xml", "left.xml", "right.xml"),
        )
        old_log = directory / "stereo" / "run-log-stereo_tri-01-01-0000-1.txt"
        new_log = directory / "stereo" / "run-log-stereo_tri-12-31-2359-9.txt"
        new_log.write_text(
            self.LOG_TEMPLATE.format(
                command="stereo_tri left.tif right.tif left.xml right.xml stereo/run"
            )
        )
        os.utime(old_log, (1000, 1000))
        os.utime(new_log, (2000, 2000))
        files = camera_files_from_stereo_run(str(directory), "stereo")
        assert [os.path.basename(f) for f in files] == ["left.xml", "right.xml"]

    def test_fewer_than_two_cameras_returns_none(self, tmp_path):
        # A mixed .xml/.json-camera command resolves a single metadata file;
        # geometry needs a pair, so the caller must fall back.
        directory = self._make_run(
            tmp_path,
            "stereo_tri left.tif right.tif "
            "left.xml ba/right.adjusted_state.json stereo/run",
            cameras=("left.xml",),
        )
        assert camera_files_from_stereo_run(str(directory), "stereo") is None

    def test_real_fixture_csm_cameras_return_none(self):
        # The committed stereo fixture's command uses CSM .json cameras.
        assert camera_files_from_stereo_run("tests/test_data", "stereo") is None

    def test_mvs_fixture_missing_cameras_return_none(self):
        # The mvs fixture's log names camera XMLs that are not on disk.
        assert camera_files_from_stereo_run("tests/test_data", "mvs/stereo") is None


class TestStereoGeometryPlotter:
    @pytest.fixture
    def stereo_geometry_plotter(self):
        # add_basemap=False keeps the test offline; a True default would fetch
        # contextily tiles from a live server and hang/flake in CI.
        stereo_geometry_plotter = StereoGeometryPlotter(
            directory="tests/test_data",
            add_basemap=False,
        )
        return stereo_geometry_plotter

    @pytest.fixture
    def stereo_geometry_plotter_tiled(self):
        stereo_geometry_plotter = StereoGeometryPlotter(
            directory="tests/test_data/tiled_xmls",
            add_basemap=False,
        )
        return stereo_geometry_plotter

    def test_composes_parser_not_inherits(self, stereo_geometry_plotter):
        # The plotter should compose a parser rather than inherit from it
        assert not isinstance(stereo_geometry_plotter, StereopairMetadataParser)
        assert isinstance(stereo_geometry_plotter.parser, StereopairMetadataParser)

    def test_stereo_geom_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.stereo_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_stereo_geom_plot_tiled(self, stereo_geometry_plotter_tiled):
        try:
            stereo_geometry_plotter_tiled.stereo_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_satellite_position_orientation_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.satellite_position_orientation_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_two_scene_single_figure(self, stereo_geometry_plotter, tmp_path):
        # Two scenes -> exactly one figure, named as given (unchanged behavior).
        saved = stereo_geometry_plotter.stereo_geom_plot(
            save_dir=str(tmp_path), fig_fn="geom.png"
        )
        assert saved == ["geom.png"]
        assert (tmp_path / "geom.png").exists()


class TestStereoGeometryPlotterNScene:
    """More than two scenes: overview figure + one figure per pair."""

    THREE_SCENES = [
        "tests/test_data/10300100D0772D00.r100.xml",
        "tests/test_data/10300100D12D7400.r100.xml",
        "tests/test_data/tiled_xmls/10200100A1865800.r100.xml",
    ]

    @pytest.fixture
    def plotter_n(self):
        return StereoGeometryPlotter(inputs=self.THREE_SCENES, add_basemap=False)

    def test_emits_overview_plus_one_figure_per_pair(self, plotter_n, tmp_path):
        # 3 scenes -> 1 overview + 3 pair figures (3-choose-2).
        saved = plotter_n.stereo_geom_plot(save_dir=str(tmp_path), fig_fn="geom.png")
        assert len(saved) == 4
        assert "geom_overview.png" in saved
        pair_files = [f for f in saved if f != "geom_overview.png"]
        assert len(pair_files) == 3
        for fn in saved:
            assert (tmp_path / fn).exists()

    def test_pair_filenames_keyed_by_catid(self, plotter_n, tmp_path):
        saved = plotter_n.stereo_geom_plot(save_dir=str(tmp_path), fig_fn="geom.png")
        catids = ["10300100D0772D00", "10300100D12D7400", "10200100A1865800"]
        pair_files = [f for f in saved if f != "geom_overview.png"]
        # Every pair file is named with both scenes' CATIDs (no pairN fallback,
        # since these scenes all carry CATIDs).
        for fn in pair_files:
            present = [c for c in catids if c in fn]
            assert len(present) == 2

    def test_single_scene_raises(self, tmp_path):
        plotter = StereoGeometryPlotter(
            inputs=["tests/test_data/10300100D0772D00.r100.xml"], add_basemap=False
        )
        with pytest.raises(ValueError, match="at least two scenes"):
            plotter.stereo_geom_plot(save_dir=str(tmp_path), fig_fn="geom.png")


class TestStereoGeometryPlotterPleiades:
    """Pléiades DIMAP inputs: no scan/TDI in labels, no covariance in plots."""

    @pytest.fixture
    def plotter(self):
        return StereoGeometryPlotter(
            directory="tests/test_data/pleiades",
            add_basemap=False,
        )

    def test_scene_string_omits_scan_and_tdi(self, plotter):
        p = plotter.parser.get_pair_dict()
        scene_string = plotter.get_scene_string(p)
        assert "ID:" in scene_string and "GSD:" in scene_string
        # DIMAP has no scan direction or TDI level; the label omits them
        # instead of printing scan:None / crashing on tdi=%i.
        assert "scan:" not in scene_string
        assert "tdi:" not in scene_string

    def test_stereo_geom_plot(self, plotter):
        try:
            plotter.stereo_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_satellite_position_orientation_plot_without_covariance(self, plotter):
        # DIMAP provides no ephemeris/attitude covariance; the plot must fall
        # back to plain position markers and an annotated covariance panel.
        try:
            plotter.satellite_position_orientation_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
