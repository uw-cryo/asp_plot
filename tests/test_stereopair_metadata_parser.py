import numpy as np
import pandas as pd
import pytest

from asp_plot.sensors import WorldViewMetadata
from asp_plot.stereopair_metadata_parser import (
    StereopairMetadataParser,
    get_asymmetry_angle,
    get_bh_ratio,
    get_bie_angle,
    get_convergence_angle,
)


class TestStereopairMetadataParser:
    @pytest.fixture
    def parser(self):
        return StereopairMetadataParser(directory="tests/test_data")

    @pytest.fixture
    def parser_tiled(self):
        return StereopairMetadataParser(directory="tests/test_data/tiled_xmls")

    def test_detects_sensor_reader(self, parser):
        # Parser is sensor-agnostic and delegates to a detected reader
        assert isinstance(parser.reader, WorldViewMetadata)

    def test_inputs_file_list_matches_directory(self, parser):
        # Building from an explicit XML file list (geom_plot *.XML) yields the
        # same pair geometry as pointing at the directory.
        files = [
            "tests/test_data/10300100D0772D00.r100.xml",
            "tests/test_data/10300100D12D7400.r100.xml",
        ]
        parser_from_files = StereopairMetadataParser(inputs=files)
        assert isinstance(parser_from_files.reader, WorldViewMetadata)
        assert parser_from_files.get_pair_dict()["conv_ang"] == pytest.approx(
            parser.get_pair_dict()["conv_ang"]
        )

    def test_requires_directory_or_inputs(self):
        with pytest.raises(ValueError, match="either a directory or inputs"):
            StereopairMetadataParser()

    def test_image_list_delegates_to_reader(self, parser):
        assert parser.image_list is parser.reader.image_list
        assert len(parser.image_list) > 0

    def test_missing_metadata_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No supported sensor metadata"):
            StereopairMetadataParser(directory=str(tmp_path))

    def test_get_pair_utm_epsg(self, parser):
        utm_epsg = parser.get_pair_utm_epsg()
        assert isinstance(utm_epsg, int)
        assert 32601 <= utm_epsg <= 32760

    def test_get_intersection_bounds_latlon(self, parser):
        bounds = parser.get_intersection_bounds()
        assert len(bounds) == 4
        min_lon, min_lat, max_lon, max_lat = bounds
        assert min_lon < max_lon
        assert min_lat < max_lat
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90

    def test_get_intersection_bounds_projected(self, parser):
        utm_epsg = parser.get_pair_utm_epsg()
        bounds = parser.get_intersection_bounds(epsg=utm_epsg)
        min_x, min_y, max_x, max_y = bounds
        assert min_x < max_x
        assert min_y < max_y
        # UTM easting/northing should be large values (not lon/lat)
        assert min_x > 100000

    def test_get_scene_bounds(self, parser):
        bounds = parser.get_scene_bounds()
        assert len(bounds) == 4
        min_lon, min_lat, max_lon, max_lat = bounds
        assert min_lon < max_lon
        assert min_lat < max_lat
        # Bounds should be valid lon/lat
        assert -180 <= min_lon <= 180
        assert -90 <= min_lat <= 90

    def test_att_df_in_catid_dict(self, parser):
        catid_dicts = parser.get_catid_dicts()
        for d in catid_dicts:
            assert "att_df" in d
            assert isinstance(d["att_df"], pd.DataFrame)
            assert len(d["att_df"]) > 0

    def test_pair_dict_stereo_geometry_values(self, parser):
        """Test stereo geometry values from real XML test data."""
        p = parser.get_pair_dict()

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

    def test_pair_dict_stereo_geometry_values_tiled(self, parser_tiled):
        """Test stereo geometry values from tiled XML test data."""
        p = parser_tiled.get_pair_dict()

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
