import matplotlib
import numpy as np
import pandas as pd
import pytest

from asp_plot.stereo_geometry import StereoGeometryPlotter
from asp_plot.stereopair_metadata_parser import (
    get_asymmetry_angle,
    get_bh_ratio,
    get_bie_angle,
    get_convergence_angle,
)

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

    def test_pair_dict_stereo_geometry_values(self, stereo_geometry_plotter):
        """Test stereo geometry values from real XML test data."""
        p = stereo_geometry_plotter.get_pair_dict()

        # All geometry keys should be present
        for key in ["conv_ang", "bh", "bie", "asymmetry_angle"]:
            assert key in p

        # Convergence angle: positive, reasonable for VHR stereo
        assert 0 < p["conv_ang"] < 90

        # B/H ratio: consistent with convergence angle
        assert p["bh"] == get_bh_ratio(p["conv_ang"])

        # BIE: positive, less than 90
        assert 0 < p["bie"] < 90

        # Asymmetry angle: non-negative and less than convergence angle
        assert 0 <= p["asymmetry_angle"] < p["conv_ang"]

        # Verify convergence matches direct computation from XML az/el
        az1, el1 = p["catid1_dict"]["meansataz"], p["catid1_dict"]["meansatel"]
        az2, el2 = p["catid2_dict"]["meansataz"], p["catid2_dict"]["meansatel"]
        assert p["conv_ang"] == get_convergence_angle(az1, el1, az2, el2)
        assert p["bie"] == get_bie_angle(az1, el1, az2, el2)

    def test_pair_dict_stereo_geometry_values_tiled(
        self, stereo_geometry_plotter_tiled
    ):
        """Test stereo geometry values from tiled XML test data."""
        p = stereo_geometry_plotter_tiled.get_pair_dict()

        for key in ["conv_ang", "bh", "bie", "asymmetry_angle"]:
            assert key in p

        assert 0 < p["conv_ang"] < 90
        assert 0 < p["bie"] < 90
        assert 0 <= p["asymmetry_angle"] < p["conv_ang"]


class TestStereoGeometryCalculations:
    """Direct tests for stereo geometry formulas (convergence, BIE, asymmetry).

    References:
        Jeong & Kim (2014), PE&RS 80(7), 653-662
        Jeong & Kim (2016), PE&RS 82(8), 625-633
    """

    def test_convergence_identical_directions(self):
        """Same viewing direction should give zero convergence."""
        conv = get_convergence_angle(az1=180, el1=75, az2=180, el2=75)
        assert conv == pytest.approx(0.0, abs=1e-10)

    def test_convergence_opposite_azimuths(self):
        """Symmetric pair at same elevation, 180 deg apart in azimuth."""
        conv = get_convergence_angle(az1=0, el1=70, az2=180, el2=70)
        # Off-nadir = 20 deg each, opposite sides -> convergence = 2 * off-nadir = 40
        assert conv == pytest.approx(40.0, abs=0.01)

    def test_convergence_symmetric_small_offset(self):
        """Two views at same elevation, small azimuth difference."""
        conv = get_convergence_angle(az1=170, el1=80, az2=190, el2=80)
        assert 0 < conv < 20  # small convergence

    def test_bh_from_convergence(self):
        """B/H = 2*tan(conv/2), verified for known convergence angles."""
        for conv in [10, 20, 30, 45, 60]:
            bh = get_bh_ratio(conv)
            assert bh == pytest.approx(2 * np.tan(np.deg2rad(conv / 2.0)), abs=0.01)

    def test_bie_symmetric_pair(self):
        """Symmetric pair with opposite azimuths: bisector is vertical, BIE = 90."""
        bie = get_bie_angle(az1=0, el1=75, az2=180, el2=75)
        assert bie == pytest.approx(90.0, abs=0.01)

    def test_bie_nadir(self):
        """Both views near-nadir with opposite azimuths: BIE = 90."""
        bie = get_bie_angle(az1=0, el1=89, az2=180, el2=89)
        assert bie == pytest.approx(90.0, abs=0.01)

    def test_bie_range(self):
        """BIE should be between 0 and 90 for any valid geometry."""
        for az1, el1, az2, el2 in [
            (45, 60, 225, 70),
            (90, 80, 270, 65),
            (10, 50, 200, 85),
        ]:
            bie = get_bie_angle(az1, el1, az2, el2)
            assert 0 < bie < 90

    def test_asymmetry_symmetric_pair(self):
        """Symmetric geometry about nadir should give asymmetry near zero."""
        # Place two satellites symmetrically about a ground point on the equator
        # Ground point on equator at lon=0: ECEF = (R, 0, 0)
        R_earth = 6.371e6
        alt = 770e3  # ~770 km orbit
        ground = np.array([R_earth, 0, 0])
        # Two sats symmetric in the x-z plane, offset by equal angles
        angle = np.deg2rad(15)
        sat1 = np.array(
            [(R_earth + alt) * np.cos(angle), 0, (R_earth + alt) * np.sin(angle)]
        )
        sat2 = np.array(
            [(R_earth + alt) * np.cos(angle), 0, -(R_earth + alt) * np.sin(angle)]
        )
        asym = get_asymmetry_angle(sat1, sat2, ground)
        assert asym == pytest.approx(0.0, abs=0.5)

    def test_asymmetry_less_than_convergence(self):
        """For typical near-nadir geometry, asymmetry should be < convergence."""
        R_earth = 6.371e6
        alt = 770e3
        ground = np.array([R_earth, 0, 0])
        # Asymmetric pair: one near-nadir, one more off-nadir
        sat1 = np.array(
            [
                (R_earth + alt) * np.cos(np.deg2rad(5)),
                0,
                (R_earth + alt) * np.sin(np.deg2rad(5)),
            ]
        )
        sat2 = np.array(
            [
                (R_earth + alt) * np.cos(np.deg2rad(20)),
                0,
                -(R_earth + alt) * np.sin(np.deg2rad(20)),
            ]
        )
        asym = get_asymmetry_angle(sat1, sat2, ground)
        # Compute convergence from the same geometry
        q1 = ground - sat1
        q1 = q1 / np.linalg.norm(q1)
        q2 = ground - sat2
        q2 = q2 / np.linalg.norm(q2)
        conv = np.rad2deg(np.arccos(np.clip(np.dot(q1, q2), -1, 1)))
        assert asym < conv
        assert asym > 0

    def test_asymmetry_ground_point_z_matters(self):
        """Verify that correct ECEF z-coordinate affects the asymmetry angle.

        This tests the bug fix: setting ECEF z=0 (equatorial plane) instead of
        using the proper ellipsoid height gives wrong results at high latitudes.
        """
        from pyproj import Transformer

        alt = 770e3

        # Ground point at ~60N latitude -- large ECEF z
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
        ground_correct = np.array(transformer.transform(10.0, 60.0, 0.0))

        # Buggy version: zero out ECEF z
        ground_buggy = np.array([ground_correct[0], ground_correct[1], 0.0])

        # Satellites roughly overhead
        sat1 = np.array(
            [ground_correct[0] + alt * 0.1, ground_correct[1], ground_correct[2] + alt]
        )
        sat2 = np.array(
            [ground_correct[0] - alt * 0.1, ground_correct[1], ground_correct[2] + alt]
        )

        asym_correct = get_asymmetry_angle(sat1, sat2, ground_correct)
        asym_buggy = get_asymmetry_angle(sat1, sat2, ground_buggy)

        # The two should differ significantly at 60N
        assert abs(asym_correct - asym_buggy) > 1.0
