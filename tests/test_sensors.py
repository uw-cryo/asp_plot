import numpy as np
import pandas as pd
import pytest

from asp_plot.sensors import (
    SENSORS,
    SensorMetadata,
    WorldViewMetadata,
    sensor_for_directory,
)


class TestWorldViewMetadata:
    @pytest.fixture
    def reader(self):
        return WorldViewMetadata(directory="tests/test_data")

    @pytest.fixture
    def reader_tiled(self):
        return WorldViewMetadata(directory="tests/test_data/tiled_xmls")

    def test_is_sensor_metadata(self, reader):
        assert isinstance(reader, SensorMetadata)
        assert reader.name == "WorldView"

    def test_image_list_excludes_ortho(self, reader):
        assert len(reader.image_list) > 0
        assert all(not f.lower().endswith("ortho.xml") for f in reader.image_list)
        assert all(f.lower().endswith(".xml") for f in reader.image_list)

    def test_missing_xml_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Missing XML camera files"):
            WorldViewMetadata(directory=str(tmp_path))

    def test_get_scene_dicts(self, reader):
        scene_dicts = reader.get_scene_dicts()
        assert isinstance(scene_dicts, list)
        assert len(scene_dicts) == 2
        for d in scene_dicts:
            for key in ["catid", "sensor", "date", "geom", "meansataz", "meansatel"]:
                assert key in d

    def test_att_df_in_scene_dict(self, reader):
        for d in reader.get_scene_dicts():
            assert "att_df" in d
            assert isinstance(d["att_df"], pd.DataFrame)
            assert len(d["att_df"]) > 0

    def test_getAtt(self, reader):
        xml = reader.image_list[0]
        att = reader.getAtt(xml)
        assert isinstance(att, np.ndarray)
        assert att.dtype == np.float64
        assert att.shape == (3, 15)

    def test_getAtt_df(self, reader):
        xml = reader.image_list[0]
        att_df = reader.getAtt_df(xml)
        assert isinstance(att_df, pd.DataFrame)
        assert isinstance(att_df.index, pd.DatetimeIndex)
        for col in ["q1", "q2", "q3", "q4"]:
            assert col in att_df.columns
        for n in ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]:
            assert f"cov_{n}" in att_df.columns

    def test_getEphem_gdf_covariance_columns(self, reader):
        xml = reader.image_list[0]
        eph_gdf = reader.getEphem_gdf(xml)
        for n in ["11", "12", "13", "22", "23", "33"]:
            assert f"cov_{n}" in eph_gdf.columns
        for old_name in ["x_cov", "y_cov", "z_cov", "dx_cov", "dy_cov", "dz_cov"]:
            assert old_name not in eph_gdf.columns

    def test_get_scene_dicts_tiled(self, reader_tiled):
        scene_dicts = reader_tiled.get_scene_dicts()
        assert len(scene_dicts) == 2


class TestSensorDetection:
    def test_detect_worldview(self):
        assert WorldViewMetadata.detect("tests/test_data") is True

    def test_detect_empty_dir(self, tmp_path):
        assert WorldViewMetadata.detect(str(tmp_path)) is False

    def test_registry_contains_worldview(self):
        assert WorldViewMetadata in SENSORS

    def test_sensor_for_directory_returns_reader(self):
        reader = sensor_for_directory("tests/test_data")
        assert isinstance(reader, WorldViewMetadata)
        assert isinstance(reader, SensorMetadata)

    def test_sensor_for_directory_no_match_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No supported sensor metadata"):
            sensor_for_directory(str(tmp_path))
