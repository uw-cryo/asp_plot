import matplotlib
import pytest

from asp_plot.stereo import StereoPlotter

matplotlib.use("Agg")


class TestStereoPlotter:
    @pytest.fixture
    def stereo_plotter(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            dem_gsd=1,
            reference_dem="tests/test_data/ref_dem.tif",
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_no_ref_dem(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            dem_gsd=1,
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_no_gsd(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            reference_dem="tests/test_data/ref_dem.tif",
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_dem_fn(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            reference_dem="tests/test_data/ref_dem.tif",
            dem_fn="date_time_left_right_1m-DEM.tif",
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_without_intersection_error(self, stereo_plotter):
        stereo_plotter.intersection_error_fn = None
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

    def test_plot_detailed_hillshade(self, stereo_plotter):
        try:
            stereo_plotter.plot_detailed_hillshade(subset_km=10)
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_detailed_hillshade_no_intersection_error(
        self, stereo_plotter_without_intersection_error
    ):
        try:
            stereo_plotter_without_intersection_error.plot_detailed_hillshade()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_instantiate_without_reference_dem(self, stereo_plotter_no_ref_dem):
        assert stereo_plotter_no_ref_dem.reference_dem is not None

    def test_instantiate_without_gsd(self, stereo_plotter_no_gsd):
        assert stereo_plotter_no_gsd.dem_fn is not None

    def test_instantiate_with_dem_fn(self, stereo_plotter_dem_fn):
        assert stereo_plotter_dem_fn.dem_fn is not None
