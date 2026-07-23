import shutil

import matplotlib
import pytest

from asp_plot.scenes import SceneFiles, ScenePlotter

matplotlib.use("Agg")


class TestSceneFiles:
    """File discovery is isolated in SceneFiles, separate from plotting."""

    @pytest.fixture
    def files(self):
        return SceneFiles(directory="tests/test_data", stereo_directory="stereo")

    def test_discovers_sub_scenes(self, files):
        assert files.left_scene_sub_fn is not None
        assert files.right_scene_sub_fn is not None

    def test_attribution_flag(self, files):
        assert files.attribution == "Vantor"


class TestScenePlotter:
    @pytest.fixture
    def scene_plotter(self):
        scene_plotter = ScenePlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
        )
        return scene_plotter

    def test_attribution_detection(self, scene_plotter):
        """Test that ScenePlotter detects Vantor satellite from test data XMLs."""
        assert scene_plotter.attribution == "Vantor"

    def test_composes_scene_files(self, scene_plotter):
        """ScenePlotter delegates discovery to a SceneFiles instance."""
        assert isinstance(scene_plotter.files, SceneFiles)
        assert scene_plotter.left_scene_sub_fn == scene_plotter.files.left_scene_sub_fn

    def test_plot_scenes(self, scene_plotter):
        try:
            scene_plotter.plot_scenes()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")


class TestScenePlotterMissingScenes:
    """A multi-view stereo directory keeps its sub-sampled scenes in run-pair*/
    subdirectories, so the top level has none: plot_scenes must draw "missing"
    placeholders instead of crashing (#160 tracks per-pair rendering)."""

    @pytest.fixture
    def scene_plotter(self):
        return ScenePlotter(
            directory="tests/test_data",
            stereo_directory="does_not_exist",
        )

    def test_scene_files_are_none(self, scene_plotter):
        assert scene_plotter.left_scene_sub_fn is None
        assert scene_plotter.right_scene_sub_fn is None

    def test_plot_scenes_placeholder(self, scene_plotter):
        try:
            scene_plotter.plot_scenes()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")


class TestScenePlotterMultiview:
    """A multi-view run's scenes are rendered one figure per run-pair*/ pair
    (issue #160)."""

    @pytest.fixture
    def scene_plotter(self):
        return ScenePlotter(
            directory="tests/test_data",
            stereo_directory="mvs/stereo",
            title="Input Scenes",
        )

    def test_top_level_scenes_absent(self, scene_plotter):
        assert scene_plotter.left_scene_sub_fn is None
        assert scene_plotter.right_scene_sub_fn is None

    def test_discovers_pairs(self, scene_plotter):
        pairs = scene_plotter.pairs
        assert [p.number for p in pairs] == [1, 2]
        assert all(p.left_scene_sub_fn is not None for p in pairs)
        assert all(p.right_scene_sub_fn is not None for p in pairs)
        assert pairs[0].label == "Pair 1: out-Band3N.tif ↔ out-Band3B.tif"

    def test_plot_scenes_per_pair(self, scene_plotter, tmp_path):
        saved = scene_plotter.plot_scenes(save_dir=str(tmp_path), fig_fn="scenes.png")
        assert saved == ["scenes_pair1.png", "scenes_pair2.png"]
        for fn in saved:
            assert (tmp_path / fn).exists()

    def test_plot_scenes_returns_single_figure_for_standard_run(self, tmp_path):
        plotter = ScenePlotter(directory="tests/test_data", stereo_directory="stereo")
        saved = plotter.plot_scenes(save_dir=str(tmp_path), fig_fn="scenes.png")
        assert saved == ["scenes.png"]
        assert (tmp_path / "scenes.png").exists()

    def test_pairs_win_over_stale_top_level_scenes(self, tmp_path):
        # Stale *-L_sub.tif/*-R_sub.tif left behind by an earlier two-image
        # run must not suppress per-pair rendering: the run-pair*/ layout is
        # the multi-view marker, matching StereoFiles.
        stereo_dir = tmp_path / "stereo"
        shutil.copytree("tests/test_data/mvs/stereo", stereo_dir)
        shutil.copy(
            stereo_dir / "run-pair1" / "1-L_sub.tif", stereo_dir / "run-L_sub.tif"
        )
        shutil.copy(
            stereo_dir / "run-pair1" / "1-R_sub.tif", stereo_dir / "run-R_sub.tif"
        )
        plotter = ScenePlotter(directory=str(tmp_path), stereo_directory="stereo")
        assert [p.number for p in plotter.pairs] == [1, 2]
        saved = plotter.plot_scenes(save_dir=str(tmp_path), fig_fn="scenes.png")
        assert saved == ["scenes_pair1.png", "scenes_pair2.png"]
