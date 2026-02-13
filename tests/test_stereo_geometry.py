import matplotlib
import numpy as np
import pandas as pd
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

    def test_getAtt(self, stereo_geometry_plotter):
        xml = stereo_geometry_plotter.image_list[0]
        att = stereo_geometry_plotter.getAtt(xml)
        assert isinstance(att, np.ndarray)
        assert att.dtype == np.float64
        assert att.shape == (3, 15)

    def test_getAtt_df(self, stereo_geometry_plotter):
        xml = stereo_geometry_plotter.image_list[0]
        att_df = stereo_geometry_plotter.getAtt_df(xml)
        assert isinstance(att_df, pd.DataFrame)
        assert isinstance(att_df.index, pd.DatetimeIndex)
        for col in ["q1", "q2", "q3", "q4"]:
            assert col in att_df.columns
        for n in ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]:
            assert f"cov_{n}" in att_df.columns

    def test_getEphem_gdf_covariance_columns(self, stereo_geometry_plotter):
        xml = stereo_geometry_plotter.image_list[0]
        eph_gdf = stereo_geometry_plotter.getEphem_gdf(xml)
        for n in ["11", "12", "13", "22", "23", "33"]:
            assert f"cov_{n}" in eph_gdf.columns
        for old_name in ["x_cov", "y_cov", "z_cov", "dx_cov", "dy_cov", "dz_cov"]:
            assert old_name not in eph_gdf.columns

    def test_att_df_in_catid_dict(self, stereo_geometry_plotter):
        catid_dicts = stereo_geometry_plotter.get_catid_dicts()
        for d in catid_dicts:
            assert "att_df" in d
            assert isinstance(d["att_df"], pd.DataFrame)
            assert len(d["att_df"]) > 0

    def test_satellite_position_orientation_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.satellite_position_orientation_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
