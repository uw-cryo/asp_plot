import matplotlib
import pytest

from asp_plot.stereo_geometry import StereoGeometryPlotter
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser

matplotlib.use("Agg")


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

    def test_dg_geom_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_dg_geom_plot_tiled(self, stereo_geometry_plotter_tiled):
        try:
            stereo_geometry_plotter_tiled.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_satellite_position_orientation_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.satellite_position_orientation_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
