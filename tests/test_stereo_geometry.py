import matplotlib
import pytest

from asp_plot.stereo_geometry import StereoGeometryPlotter

matplotlib.use("Agg")


class TestStereoGeometryPlotter:
    @pytest.fixture
    def stereo_geometry_plotter(self):
        stereo_geometry_plotter = StereoGeometryPlotter(
            directory="tests/test_data",
        )
        return stereo_geometry_plotter

    @pytest.fixture
    def stereo_geometry_plotter_tiled(self):
        stereo_geometry_plotter = StereoGeometryPlotter(
            directory="tests/test_data/tiled_xmls",
        )
        return stereo_geometry_plotter

    def test_dg_geom_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_get_pair_utm_epsg(self, stereo_geometry_plotter):
        utm_epsg = stereo_geometry_plotter.get_pair_utm_epsg()
        assert isinstance(utm_epsg, int)
        assert 32601 <= utm_epsg <= 32760

    def test_get_intersection_bounds_latlon(self, stereo_geometry_plotter):
        bounds = stereo_geometry_plotter.get_intersection_bounds()
        assert len(bounds) == 4
        min_lon, min_lat, max_lon, max_lat = bounds
        assert min_lon < max_lon
        assert min_lat < max_lat
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90

    def test_get_intersection_bounds_projected(self, stereo_geometry_plotter):
        utm_epsg = stereo_geometry_plotter.get_pair_utm_epsg()
        bounds = stereo_geometry_plotter.get_intersection_bounds(epsg=utm_epsg)
        min_x, min_y, max_x, max_y = bounds
        assert min_x < max_x
        assert min_y < max_y
        # UTM easting/northing should be large values (not lon/lat)
        assert min_x > 100000

    def test_get_scene_bounds(self, stereo_geometry_plotter):
        bounds = stereo_geometry_plotter.get_scene_bounds()
        assert len(bounds) == 4
        min_lon, min_lat, max_lon, max_lat = bounds
        assert min_lon < max_lon
        assert min_lat < max_lat
        # Bounds should be valid lon/lat
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90

    def test_dg_geom_plot_tiled(self, stereo_geometry_plotter_tiled):
        try:
            stereo_geometry_plotter_tiled.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
