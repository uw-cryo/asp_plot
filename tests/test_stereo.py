import pytest
from asp_plot.stereo import StereoPlotter
import matplotlib

matplotlib.use("Agg")


class TestStereoPlotter:
    @pytest.fixture
    def stereo_plotter(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            reference_dem="tests/test_data/ref_dem.tif",
            out_dem_gsd=1,
            title="Stereo Results",
        )
        return stereo_plotter

    def test_plot_match_points(self, stereo_plotter):
        try:
            stereo_plotter.plot_match_points()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_disparity(self, stereo_plotter):
        try:
            stereo_plotter.plot_disparity(unit="meters")
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_dem_results(self, stereo_plotter):
        try:
            stereo_plotter.plot_dem_results()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
