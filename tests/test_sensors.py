import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asp_plot.sensors import (
    SENSORS,
    SensorMetadata,
    WorldViewMetadata,
    resolve_xml_inputs,
    sensor_for_directory,
    sensor_for_inputs,
)

# The two committed single-scene WorldView camera XMLs at the top level of
# tests/test_data (one *.r100.xml per CATID, no tiles).
TEST_DATA_DIR = Path("tests/test_data")
CAM_A = TEST_DATA_DIR / "10300100D0772D00.r100.xml"
CAM_B = TEST_DATA_DIR / "10300100D12D7400.r100.xml"


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


class TestWorldViewDiscovery:
    """Shallow-first XML discovery and non-camera XML exclusion."""

    def test_finds_xml_nested_several_dirs_deep(self, tmp_path):
        # Real deliveries nest the camera XML well below the directory handed in.
        nested = tmp_path / "order" / "DVD_VOL_1" / "order" / "scene_PAN"
        nested.mkdir(parents=True)
        shutil.copy(CAM_A, nested / "camera.xml")

        found = WorldViewMetadata._discover_xmls(str(tmp_path))
        assert [Path(f).name for f in found] == ["camera.xml"]

    def test_top_level_takes_precedence_over_nested(self, tmp_path):
        # A flat delivery / processing dir keeps its camera XMLs at the top
        # level, so discovery uses those and does NOT descend into unrelated
        # subdirectories (which would change report behavior).
        shutil.copy(CAM_A, tmp_path / "top.xml")
        nested = tmp_path / "subdir"
        nested.mkdir()
        shutil.copy(CAM_B, nested / "nested.xml")

        found = WorldViewMetadata._discover_xmls(str(tmp_path))
        assert [Path(f).name for f in found] == ["top.xml"]

    def test_non_recursive_ignores_nested(self, tmp_path):
        nested = tmp_path / "scene_PAN"
        nested.mkdir()
        shutil.copy(CAM_A, nested / "camera.xml")

        assert WorldViewMetadata._discover_xmls(str(tmp_path), recursive=False) == []

    def test_excludes_readme_and_ortho(self, tmp_path):
        pan = tmp_path / "scene_PAN"
        pan.mkdir()
        shutil.copy(CAM_A, pan / "camera.xml")
        # Decoys that ship alongside camera XMLs and must be ignored by name.
        (tmp_path / "500647760070_01_README.XML").write_text("<README/>")
        (pan / "scene-ortho.xml").write_text("<isd/>")

        found = WorldViewMetadata._discover_xmls(str(tmp_path))
        assert [Path(f).name for f in found] == ["camera.xml"]

    def test_detect_uses_recursive_discovery(self, tmp_path):
        nested = tmp_path / "a" / "b" / "scene_PAN"
        nested.mkdir(parents=True)
        shutil.copy(CAM_A, nested / "camera.xml")
        assert WorldViewMetadata.detect(str(tmp_path)) is True


class TestFlexibleInputResolution:
    """Resolving files / directories / globs into camera XMLs (geom_plot *.XML)."""

    def test_resolve_explicit_files(self):
        # The shell-expanded ``geom_plot *.XML`` case: a list of file paths.
        found = resolve_xml_inputs([str(CAM_A), str(CAM_B)])
        assert found == sorted([str(CAM_A), str(CAM_B)])

    def test_resolve_directory(self):
        # A bare directory is discovered the same way as the directory API.
        found = resolve_xml_inputs(str(TEST_DATA_DIR))
        assert set(found) == {str(CAM_A), str(CAM_B)}

    def test_resolve_glob_pattern(self):
        found = resolve_xml_inputs(str(TEST_DATA_DIR / "*.r100.xml"))
        assert set(found) == {str(CAM_A), str(CAM_B)}

    def test_resolve_single_string_not_iterated_per_char(self):
        # A lone path string must be treated as one path, not a char iterable.
        found = resolve_xml_inputs(str(CAM_A))
        assert found == [str(CAM_A)]

    def test_resolve_mixed_inputs_deduplicated(self, tmp_path):
        # A file, a directory, and a glob that all reference overlapping XMLs
        # collapse to a de-duplicated, sorted set.
        shutil.copy(CAM_A, tmp_path / "a.xml")
        shutil.copy(CAM_B, tmp_path / "b.xml")
        found = resolve_xml_inputs(
            [
                str(tmp_path / "a.xml"),
                str(tmp_path),  # also discovers a.xml and b.xml
                str(tmp_path / "*.xml"),  # again
            ]
        )
        assert found == sorted([str(tmp_path / "a.xml"), str(tmp_path / "b.xml")])

    def test_resolve_finds_nested_xml(self, tmp_path):
        # A directory input still descends into deeply-nested deliveries.
        nested = tmp_path / "order" / "scene_PAN"
        nested.mkdir(parents=True)
        shutil.copy(CAM_A, nested / "camera.xml")
        found = resolve_xml_inputs(str(tmp_path))
        assert [Path(f).name for f in found] == ["camera.xml"]

    def test_resolve_missing_input_skipped(self, tmp_path, caplog):
        with caplog.at_level("WARNING"):
            found = resolve_xml_inputs(
                [str(CAM_A), str(tmp_path / "does_not_exist.xml")]
            )
        assert found == [str(CAM_A)]
        assert "does not exist" in caplog.text

    def test_sensor_for_inputs_returns_reader(self):
        reader = sensor_for_inputs([str(CAM_A), str(CAM_B)])
        assert isinstance(reader, WorldViewMetadata)
        assert set(reader.image_list) == {str(CAM_A), str(CAM_B)}

    def test_sensor_for_inputs_no_xml_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No XML files found"):
            sensor_for_inputs([str(tmp_path / "nope.xml")])

    def test_reader_from_image_list_filters_non_camera(self, tmp_path):
        # README/ortho decoys handed in explicitly are still dropped.
        readme = tmp_path / "500647760070_01_README.XML"
        readme.write_text("<README/>")
        reader = WorldViewMetadata(image_list=[str(CAM_A), str(readme)])
        assert reader.image_list == [str(CAM_A)]

    def test_reader_requires_directory_or_image_list(self):
        with pytest.raises(ValueError, match="either a directory or an image_list"):
            WorldViewMetadata()


class TestWorldViewSceneGrouping:
    """Grouping discovered XMLs into scenes by CATID read from content."""

    @staticmethod
    def _scene_with_catid(src, dst, new_catid):
        """Copy ``src`` to ``dst`` with its CATID rewritten to ``new_catid``."""
        # CAM_A's CATID is its filename stem; rewrite every occurrence so the
        # copy reads as a distinct scene.
        text = Path(src).read_text().replace("10300100D0772D00", new_catid)
        Path(dst).write_text(text)

    def test_distinct_single_tile_scenes_not_mosaicked(self, tmp_path):
        # Three distinct single-tile scenes in one flat directory must NOT be
        # treated as tiles of one scene just because there are more than two.
        shutil.copy(CAM_A, tmp_path / "a.xml")
        shutil.copy(CAM_B, tmp_path / "b.xml")
        self._scene_with_catid(CAM_A, tmp_path / "c.xml", "10300100DEADBE00")

        reader = WorldViewMetadata(directory=str(tmp_path))
        catid_xmls = reader.get_catid_xmls()

        assert set(catid_xmls) == {
            "10300100D0772D00",
            "10300100D12D7400",
            "10300100DEADBE00",
        }
        # Each scene maps to one of the untouched inputs ...
        assert {Path(v).name for v in catid_xmls.values()} == {
            "a.xml",
            "b.xml",
            "c.xml",
        }
        # ... and no dg_mosaic output was produced.
        assert not list(tmp_path.glob("*_asp_plot_dg_mosaic*"))

    def test_skips_xml_without_catid(self, tmp_path, caplog):
        shutil.copy(CAM_A, tmp_path / "a.xml")
        shutil.copy(CAM_B, tmp_path / "b.xml")
        # A non-camera XML whose name does not match readme/ortho, so it passes
        # discovery and must be skipped by the content (CATID) check.
        (tmp_path / "sidecar.xml").write_text("<metadata><note>hi</note></metadata>")

        reader = WorldViewMetadata(directory=str(tmp_path))
        with caplog.at_level("WARNING"):
            catid_xmls = reader.get_catid_xmls()

        assert set(catid_xmls) == {"10300100D0772D00", "10300100D12D7400"}
        assert "without a CATID" in caplog.text

    def test_all_xmls_without_catid_raises(self, tmp_path):
        (tmp_path / "sidecar.xml").write_text("<metadata/>")
        reader = WorldViewMetadata(directory=str(tmp_path))
        with pytest.raises(ValueError, match="No XML camera files with a CATID"):
            reader.get_catid_xmls()

    def test_lone_r100_delivery_used_as_is(self):
        # A scene delivered as a single *.r100.xml is the camera itself and must
        # not be dropped as a regenerable mosaic intermediate.
        reader = WorldViewMetadata(directory=str(TEST_DATA_DIR))
        catid_xmls = reader.get_catid_xmls()
        assert set(catid_xmls) == {"10300100D0772D00", "10300100D12D7400"}
        assert all(v.endswith(".r100.xml") for v in catid_xmls.values())

    def test_tiled_scene_reuses_existing_mosaic(self):
        # Raw tiles + a pre-existing mosaic: grouped to one mosaic per CATID
        # without invoking dg_mosaic (the committed mosaic output is reused).
        reader = WorldViewMetadata(directory="tests/test_data/tiled_xmls")
        catid_xmls = reader.get_catid_xmls()
        assert set(catid_xmls) == {"10200100A1865800", "10200100A37C1C00"}
        assert all(
            v.endswith("_asp_plot_dg_mosaic.r100.xml") for v in catid_xmls.values()
        )
