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
