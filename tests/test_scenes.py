import pytest
from asp_plot.scenes import ScenePlotter
import matplotlib

matplotlib.use("Agg")


class TestScenePlotter:
    @pytest.fixture
    def scene_plotter(self):
        left_image_fn = "tests/test_data/stereo/mini-L_sub.tif"
        right_image_fn = "tests/test_data/stereo/mini-R_sub.tif"
        return ScenePlotter(left_image_fn, right_image_fn)

    def test_plot_orthos(self, scene_plotter):
        try:
            scene_plotter.plot_orthos(title="Orthos")
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
