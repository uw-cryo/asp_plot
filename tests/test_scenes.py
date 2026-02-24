import matplotlib
import pytest

from asp_plot.scenes import ScenePlotter

matplotlib.use("Agg")


class TestScenePlotter:
    @pytest.fixture
    def scene_plotter(self):
        scene_plotter = ScenePlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
        )
        return scene_plotter

    def test_is_vantor_detection(self, scene_plotter):
        """Test that ScenePlotter detects Vantor satellite from test data XMLs."""
        assert scene_plotter.is_vantor is True

    def test_plot_scenes(self, scene_plotter):
        try:
            scene_plotter.plot_scenes()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
