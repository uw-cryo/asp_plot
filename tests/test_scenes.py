import matplotlib
import pytest

from asp_plot.scenes import SceneGeometryPlotter, ScenePlotter

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

    @pytest.fixture
    def scene_geometry_plotter(self):
        scene_geometry_plotter = SceneGeometryPlotter(
            directory="tests/test_data",
        )
        return scene_geometry_plotter

    def test_plot_orthos(self, scene_plotter):
        try:
            scene_plotter.plot_orthos()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_dg_geom_plot(self, scene_geometry_plotter):
        try:
            scene_geometry_plotter.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
