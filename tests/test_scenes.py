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
            title="Mapprojected Scenes",
        )
        return scene_plotter

    def test_plot_orthos(self, scene_plotter):
        try:
            scene_plotter.plot_orthos()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
