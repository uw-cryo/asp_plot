import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asp_plot.sensors import (
    SENSORS,
    PleiadesMetadata,
    PrismMetadata,
    SensorMetadata,
    Spot5Metadata,
    WorldViewMetadata,
)
from asp_plot.sensors import dimap as dimap_module
from asp_plot.sensors import dimap_v1 as dimap_v1_module
from asp_plot.sensors import resolve_xml_inputs, sensor_for_directory, sensor_for_inputs
from asp_plot.sensors.dimap import SUPPORTED_DIMAP_PROFILES

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

    @staticmethod
    def _scene_without_catid(src, dst):
        """Copy ``src`` to ``dst`` with every CATID tag removed.

        The result still passes the DG content check (root ``<isd>`` with
        IMD/EPH/ATT blocks) so it exercises the later CATID-grouping skip
        path rather than being dropped at discovery.
        """
        text = re.sub(r"<CATID>[^<]*</CATID>", "", Path(src).read_text())
        Path(dst).write_text(text)

    def test_skips_xml_without_catid(self, tmp_path, caplog):
        shutil.copy(CAM_A, tmp_path / "a.xml")
        shutil.copy(CAM_B, tmp_path / "b.xml")
        # A DG-shaped XML without a CATID passes the content check at
        # discovery and must be skipped by the CATID grouping. (A plain
        # non-camera sidecar XML no longer reaches this point at all — the
        # #162 content check drops it at discovery.)
        self._scene_without_catid(CAM_A, tmp_path / "sidecar.xml")

        reader = WorldViewMetadata(directory=str(tmp_path))
        with caplog.at_level("WARNING"):
            catid_xmls = reader.get_catid_xmls()

        assert set(catid_xmls) == {"10300100D0772D00", "10300100D12D7400"}
        assert "without a CATID" in caplog.text

    def test_all_xmls_without_catid_raises(self, tmp_path):
        self._scene_without_catid(CAM_A, tmp_path / "sidecar.xml")
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


# Trimmed Pléiades Neo DIMAP fixtures (Airbus Marseille sample data; ephemeris
# and attitude lists truncated to 8 entries). Fore + aft scenes of a tri-stereo
# (~21.7 deg convergence) plus one RPC sidecar that must be filtered out.
PLEIADES_DIR = TEST_DATA_DIR / "pleiades"
DIM_FORE = PLEIADES_DIR / "DIM_PNEO3_202111071029126_PAN_trimmed.XML"
DIM_AFT = PLEIADES_DIR / "DIM_PNEO3_202111071029456_PAN_trimmed.XML"
RPC_FORE = PLEIADES_DIR / "RPC_PNEO3_202111071029126_PAN_trimmed.XML"


class TestPleiadesMetadata:
    @pytest.fixture
    def reader(self):
        return PleiadesMetadata(directory=str(PLEIADES_DIR))

    def test_is_sensor_metadata(self, reader):
        assert isinstance(reader, SensorMetadata)
        assert reader.name == "Pleiades"

    def test_image_list_excludes_rpc(self, reader):
        # Both DIM product XMLs found; the RPC sidecar (same DIMAP root tag,
        # METADATA_SUBPROFILE of RPC) is filtered out.
        assert len(reader.image_list) == 2
        assert all("DIM_PNEO3" in f for f in reader.image_list)

    def test_missing_xml_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Missing DIMAP"):
            PleiadesMetadata(directory=str(tmp_path))

    def test_get_scene_dicts(self, reader):
        scene_dicts = reader.get_scene_dicts()
        assert len(scene_dicts) == 2
        for d in scene_dicts:
            for key in [
                "catid",
                "sensor",
                "date",
                "geom",
                "meansataz",
                "meansatel",
                "meanoffnadirviewangle",
                "meanintrackviewangle",
                "meancrosstrackviewangle",
                "meanproductgsd",
                "meansunaz",
                "meansunel",
                "cloudcover",
            ]:
                assert key in d
            assert d["sensor"] == "PNEO3"
            # DIMAP has no scan direction / TDI level.
            assert d["scandir"] is None
            assert d["tdi"] is None
            assert d["geom"].is_valid

    def test_scene_values_fore(self, reader):
        d = [s for s in reader.get_scene_dicts() if "202111071029126" in s["catid"]][0]
        assert d["catid"] == "PNEO3_202111071029126_PAN_SEN"
        assert d["date"].year == 2021 and d["date"].month == 11
        # Fore scene of the tri-stereo: ~22 deg incidence -> ~68 deg elevation.
        assert d["meansatel"] == pytest.approx(68, abs=1)
        assert d["meanproductgsd"] == pytest.approx(0.33, abs=0.05)
        assert d["meansunel"] == pytest.approx(29, abs=1)
        assert d["cloudcover"] == 0.0

    def test_eph_gdf(self, reader):
        d = reader.get_scene_dicts()[0]
        eph_gdf = d["eph_gdf"]
        assert isinstance(eph_gdf.index, pd.DatetimeIndex)
        assert len(eph_gdf) == 8  # trimmed fixture
        for col in ["x", "y", "z", "dx", "dy", "dz"]:
            assert col in eph_gdf.columns
        # Positions are ECEF meters (satellite altitude ~7000 km radius).
        radius = np.sqrt(eph_gdf["x"] ** 2 + eph_gdf["y"] ** 2 + eph_gdf["z"] ** 2)
        assert ((radius > 6.9e6) & (radius < 7.1e6)).all()
        # DIMAP provides no ephemeris covariance: columns exist but are NaN.
        for n in ["11", "12", "13", "22", "23", "33"]:
            assert eph_gdf[f"cov_{n}"].isna().all()

    def test_att_df_scalar_last_quaternions(self, reader):
        d = reader.get_scene_dicts()[0]
        att_df = d["att_df"]
        assert isinstance(att_df.index, pd.DatetimeIndex)
        assert len(att_df) == 8  # trimmed fixture
        # Unit quaternions, reordered scalar-last (Airbus Q0 lands in q4).
        norm = np.sqrt(
            att_df["q1"] ** 2
            + att_df["q2"] ** 2
            + att_df["q3"] ** 2
            + att_df["q4"] ** 2
        )
        assert np.allclose(norm, 1.0, atol=1e-6)
        for n in ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]:
            assert att_df[f"cov_{n}"].isna().all()

    def test_att_df_reorder_pinned_to_raw_fixture_values(self, reader):
        # Pin the scalar-first -> scalar-last reorder against the raw XML:
        # Airbus Q0 (the scalar part) must land in q4, and Q1..Q3 in q1..q3.
        # Guards the reorder against being "fixed" to a pass-through (the
        # ordering matches ASP's own reader, which maps Q0 to the scalar w).
        import xml.etree.ElementTree as ET

        first_quat = (
            ET.parse(DIM_FORE)
            .getroot()
            .find(".//Refined_Model/Attitudes/Quaternion_List/Quaternion")
        )
        d = [s for s in reader.get_scene_dicts() if "202111071029126" in s["catid"]][0]
        row = d["att_df"].iloc[0]
        assert row["q1"] == float(first_quat.findtext("Q1"))
        assert row["q2"] == float(first_quat.findtext("Q2"))
        assert row["q3"] == float(first_quat.findtext("Q3"))
        assert row["q4"] == float(first_quat.findtext("Q0"))

    def test_pair_geometry_matches_bundle_adjust(self):
        # Convergence angle of the fore/aft pair as measured by ASP
        # bundle_adjust on the full images is 21.7 deg; the DIMAP-derived
        # value must agree closely.
        from asp_plot.stereopair_metadata_parser import StereopairMetadataParser

        parser = StereopairMetadataParser(directory=str(PLEIADES_DIR))
        p = parser.get_pair_dict()
        assert p["conv_ang"] == pytest.approx(21.7, abs=0.3)
        assert p["bh"] == pytest.approx(0.38, abs=0.02)
        assert p["intersection_area"] > 50


class TestPleiadesDetection:
    def test_detect_pleiades_dir(self):
        assert PleiadesMetadata.detect(str(PLEIADES_DIR)) is True

    def test_detect_rejects_worldview_dir_shallow(self):
        # WorldView XMLs are not DIMAP; shallow detection must not match the
        # top-level test_data dir (the nested pleiades/ fixtures are only
        # reachable recursively).
        assert PleiadesMetadata.detect(str(TEST_DATA_DIR), recursive=False) is False

    def test_detect_empty_dir(self, tmp_path):
        assert PleiadesMetadata.detect(str(tmp_path)) is False

    def test_registry_order_pleiades_first(self):
        # Pléiades detects strictly on the DIMAP root tag while WorldView
        # matches any non-ortho XML, so Pléiades must be checked first.
        assert SENSORS.index(PleiadesMetadata) < SENSORS.index(WorldViewMetadata)

    def test_sensor_for_directory_pleiades(self):
        reader = sensor_for_directory(str(PLEIADES_DIR))
        assert isinstance(reader, PleiadesMetadata)

    def test_sensor_for_directory_worldview_top_level_wins(self):
        # tests/test_data has WorldView XMLs at the top level and DIMAP
        # fixtures nested in pleiades/: the shallow match must win.
        reader = sensor_for_directory(str(TEST_DATA_DIR))
        assert isinstance(reader, WorldViewMetadata)

    def test_sensor_for_inputs_pleiades_files(self):
        reader = sensor_for_inputs([str(DIM_FORE), str(DIM_AFT)])
        assert isinstance(reader, PleiadesMetadata)
        assert len(reader.image_list) == 2

    def test_detect_files_rejects_rpc_only(self):
        assert PleiadesMetadata.detect_files([str(RPC_FORE)]) is False

    def test_rpc_only_inputs_rejected_cleanly(self):
        # An RPC sidecar alone is not a camera model for any reader: Pléiades
        # rejects it (METADATA_SUBPROFILE is RPC, not PRODUCT) and the
        # WorldView content check rejects it too (DIMAP root, no DG blocks),
        # so the user gets the clean "no supported sensor" error instead of a
        # confusing parse failure deep in the WorldView reader (#162).
        with pytest.raises(ValueError, match="No supported sensor metadata"):
            sensor_for_inputs([str(RPC_FORE)])


# ASP gen_aster camera XML: shares the <isd> root with DigitalGlobe XMLs but
# carries none of the IMD/EPH/ATT blocks — the canonical "same container,
# different sensor" case the WorldView content check must reject (#162).
ASTER_CAM = TEST_DATA_DIR / "no_mapproj" / "out-Band3N.xml"


class TestWorldViewContentDetection:
    """#162: WorldView claims files by content (isd root + DG blocks), not name."""

    def test_accepts_real_camera_xmls(self):
        assert WorldViewMetadata._is_camera_file(str(CAM_A)) is True
        assert WorldViewMetadata._is_camera_file(str(CAM_B)) is True

    def test_rejects_arbitrary_xml(self, tmp_path):
        f = tmp_path / "unrelated.xml"
        f.write_text("<metadata><note>hi</note></metadata>")
        assert WorldViewMetadata._is_camera_file(str(f)) is False

    def test_rejects_isd_root_without_dg_blocks(self):
        # gen_aster output: <isd> root but LATTICE_POINT/SIGHT_VECTOR content.
        assert WorldViewMetadata._is_camera_file(str(ASTER_CAM)) is False

    def test_rejects_dimap_xml(self):
        assert WorldViewMetadata._is_camera_file(str(DIM_FORE)) is False
        assert WorldViewMetadata._is_camera_file(str(RPC_FORE)) is False

    def test_rejects_unparseable_file(self, tmp_path):
        f = tmp_path / "broken.xml"
        f.write_text("<isd><IMD>")
        assert WorldViewMetadata._is_camera_file(str(f)) is False

    def test_unrelated_xml_dir_raises_clean_error(self, tmp_path):
        (tmp_path / "unrelated.xml").write_text("<metadata/>")
        with pytest.raises(ValueError, match="No supported sensor metadata"):
            sensor_for_directory(str(tmp_path))

    def test_aster_dir_not_claimed(self, tmp_path):
        # A directory holding only gen_aster camera XMLs must not be claimed
        # by the WorldView reader.
        shutil.copy(ASTER_CAM, tmp_path / "out-Band3N.xml")
        assert WorldViewMetadata.detect(str(tmp_path)) is False
        with pytest.raises(ValueError, match="No supported sensor metadata"):
            sensor_for_directory(str(tmp_path))

    def test_explicit_image_list_content_filtered(self, tmp_path):
        # Content filtering also applies to explicit inputs, not only
        # directory discovery.
        f = tmp_path / "unrelated.xml"
        f.write_text("<metadata/>")
        reader = WorldViewMetadata(image_list=[str(CAM_A), str(f)])
        assert reader.image_list == [str(CAM_A)]


class TestDimapProfileGating:
    """Products from unsupported DIMAP profiles are skipped with a warning."""

    @staticmethod
    def _dimap_with_profile(dst, profile, subprofile="PRODUCT"):
        dst.write_text(
            "<Dimap_Document>"
            "<Metadata_Identification>"
            f"<METADATA_PROFILE>{profile}</METADATA_PROFILE>"
            f"<METADATA_SUBPROFILE>{subprofile}</METADATA_SUBPROFILE>"
            "</Metadata_Identification>"
            "</Dimap_Document>"
        )

    def test_supported_profiles_accepted(self, tmp_path):
        for profile in SUPPORTED_DIMAP_PROFILES:
            f = tmp_path / f"DIM_{profile}.XML"
            self._dimap_with_profile(f, profile)
            assert PleiadesMetadata._is_camera_file(str(f)) is True

    def test_unsupported_profile_rejected_with_warning(self, tmp_path, caplog):
        f = tmp_path / "DIM_UNKNOWN.XML"
        self._dimap_with_profile(f, "UNKNOWN_SENSOR")
        with caplog.at_level("WARNING"):
            assert PleiadesMetadata._is_camera_file(str(f)) is False
        assert "UNKNOWN_SENSOR" in caplog.text
        assert "#168" in caplog.text

    def test_unsupported_profile_warns_once(self, tmp_path, caplog):
        f = tmp_path / "DIM_UNKNOWN.XML"
        self._dimap_with_profile(f, "UNKNOWN_SENSOR")
        with caplog.at_level("WARNING"):
            PleiadesMetadata._is_camera_file(str(f))
            caplog.clear()
            PleiadesMetadata._is_camera_file(str(f))
        assert "UNKNOWN_SENSOR" not in caplog.text

    def test_rpc_subprofile_rejected_silently(self, tmp_path, caplog):
        f = tmp_path / "RPC_PNEO.XML"
        self._dimap_with_profile(f, "PNEO_SENSOR", subprofile="RPC")
        with caplog.at_level("WARNING"):
            assert PleiadesMetadata._is_camera_file(str(f)) is False
        assert "unsupported" not in caplog.text

    def test_real_pneo_fixture_still_accepted(self):
        assert PleiadesMetadata._is_camera_file(str(DIM_FORE)) is True


# Synthetic DIMAP fixtures for the profiles added in #168: a Pléiades 1A/1B
# product with Polynomial_Quaternions attitude (#161) and a PeruSat-1 product
# with a single Located_Geometric_Values block. Both are derived from the
# trimmed PNEO fixture and model the layouts ASP's readers (PleiadesXML.cc,
# PeruSatXML.cc) parse. They live in their own directory so the pleiades/
# fixture counts stay untouched.
DIMAP_SYNTH_DIR = TEST_DATA_DIR / "dimap_synthetic"
DIM_PHR = DIMAP_SYNTH_DIR / "DIM_PHR1A_202111071029126_PAN_synthetic.XML"
DIM_PER1 = DIMAP_SYNTH_DIR / "DIM_PER1_202111071029126_PAN_synthetic.XML"


class TestPleiades1A1BPolynomialAttitude:
    """#161: 1A/1B Polynomial_Quaternions evaluate to a tabulated att_df."""

    @pytest.fixture
    def scene(self):
        return PleiadesMetadata(image_list=[str(DIM_PHR)]).get_scene_dicts()[0]

    def test_accepted_by_content_detection(self):
        assert PleiadesMetadata._is_camera_file(str(DIM_PHR)) is True

    def test_scene_identity(self, scene):
        assert scene["sensor"] == "PHR1A"
        assert scene["catid"] == "PHR1A_202111071029126_PAN_SEN"
        assert scene["date"].year == 2021 and scene["date"].month == 11

    def test_att_df_sampled_at_ephemeris_times(self, scene):
        # The polynomial is evaluated at the ephemeris timestamps, so the
        # attitude table lines up row-for-row with the ephemeris.
        assert scene["att_df"].index.equals(scene["eph_gdf"].index)
        assert len(scene["att_df"]) == 8  # trimmed fixture

    def test_att_df_unit_quaternions(self, scene):
        att_df = scene["att_df"]
        norm = np.sqrt((att_df[["q1", "q2", "q3", "q4"]] ** 2).sum(axis=1))
        assert np.allclose(norm, 1.0, atol=1e-12)

    def test_polynomial_evaluation_pinned(self, scene):
        # Hand-computed from the fixture's OFFSET/SCALE/COEFFICIENTS with an
        # ASP-style ascending-power loop, independent of the np.polynomial
        # implementation under test: scaled_t = (seconds since midnight of
        # the start date - 37751.95) / 0.01, one cubic per component, then
        # normalized and reordered scalar-last.
        first = scene["att_df"].iloc[0]
        assert first["q1"] == pytest.approx(0.9296693569966066, rel=1e-12)
        assert first["q2"] == pytest.approx(0.12381232823473337, rel=1e-12)
        assert first["q3"] == pytest.approx(-0.3055356636094778, rel=1e-12)
        assert first["q4"] == pytest.approx(0.16441822375067466, rel=1e-12)
        last = scene["att_df"].iloc[-1]
        assert last["q1"] == pytest.approx(0.9297091923203954, rel=1e-12)
        assert last["q2"] == pytest.approx(0.12405665881524922, rel=1e-12)
        assert last["q3"] == pytest.approx(-0.30556125333626116, rel=1e-12)
        assert last["q4"] == pytest.approx(0.1639606159360475, rel=1e-12)

    def test_wrong_polynomial_degree_raises(self, tmp_path):
        f = tmp_path / "DIM_PHR_DEG2.XML"
        f.write_text(
            DIM_PHR.read_text().replace("<DEGREE>3</DEGREE>", "<DEGREE>2</DEGREE>", 1)
        )
        reader = PleiadesMetadata(image_list=[str(f)])
        with pytest.raises(ValueError, match="degree of the quaternion polynomial"):
            reader.get_scene_dicts()

    def test_missing_attitude_raises(self, tmp_path):
        text = re.sub(
            r"<Polynomial_Quaternions>.*?</Polynomial_Quaternions>",
            "",
            DIM_PHR.read_text(),
            flags=re.DOTALL,
        )
        f = tmp_path / "DIM_PHR_NOATT.XML"
        f.write_text(text)
        reader = PleiadesMetadata(image_list=[str(f)])
        with pytest.raises(ValueError, match="No attitude found"):
            reader.get_scene_dicts()


class TestPeruSatMetadata:
    """PER1_SENSOR: same DIMAP layout, single Located_Geometric_Values."""

    @pytest.fixture
    def scene(self):
        return PleiadesMetadata(image_list=[str(DIM_PER1)]).get_scene_dicts()[0]

    def test_accepted_by_content_detection(self):
        assert PleiadesMetadata._is_camera_file(str(DIM_PER1)) is True

    def test_scene_identity(self, scene):
        assert scene["sensor"] == "PER1"
        assert scene["catid"] == "PER1_202111071029126_PAN_SEN"

    def test_single_lgv_means(self, scene):
        # With one Located_Geometric_Values block, the "means" are just that
        # (center) block's values.
        assert scene["meansataz"] == pytest.approx(63.05, abs=0.01)
        assert scene["meansatel"] == pytest.approx(68.22, abs=0.01)
        assert scene["meansunel"] == pytest.approx(29.08, abs=0.01)

    def test_tabulated_attitude(self, scene):
        assert len(scene["att_df"]) == 8
        assert not scene["att_df"][["q1", "q2", "q3", "q4"]].isna().any().any()


class TestDimapSpecOnlyProfileWarning:
    """Spec-only profiles (S6/S7/PER1) warn once when parsed (#168)."""

    @pytest.fixture(autouse=True)
    def _reset_warned(self, monkeypatch):
        monkeypatch.setattr(dimap_module, "_warned_spec_only_profiles", set())

    def test_spec_only_profile_warns_once(self, caplog):
        reader = PleiadesMetadata(image_list=[str(DIM_PER1)])
        with caplog.at_level("WARNING"):
            reader.get_scene_dicts()
        assert "PER1_SENSOR" in caplog.text
        assert "ASP reader spec" in caplog.text
        assert "issues/168" in caplog.text
        caplog.clear()
        with caplog.at_level("WARNING"):
            reader.get_scene_dicts()
        assert "ASP reader spec" not in caplog.text

    def test_validated_profile_does_not_warn(self, caplog):
        reader = PleiadesMetadata(image_list=[str(DIM_FORE)])
        with caplog.at_level("WARNING"):
            reader.get_scene_dicts()
        assert "ASP reader spec" not in caplog.text

    def test_spot6_fabricated_scene(self, tmp_path, caplog):
        # SPOT 6 products share the PNEO layout apart from the profile and
        # mission strings, so fabricate one from the PNEO fixture instead of
        # committing a third near-identical 25 KB file.
        text = DIM_FORE.read_text()
        text = text.replace("PNEO_SENSOR", "S6_SENSOR")
        text = text.replace("<MISSION>PNEO</MISSION>", "<MISSION>SPOT</MISSION>")
        text = text.replace(
            "<MISSION_INDEX>3</MISSION_INDEX>", "<MISSION_INDEX>6</MISSION_INDEX>"
        )
        f = tmp_path / "DIM_SPOT6.XML"
        f.write_text(text)
        assert PleiadesMetadata._is_camera_file(str(f)) is True
        with caplog.at_level("WARNING"):
            d = PleiadesMetadata(image_list=[str(f)]).get_scene_dicts()[0]
        assert d["sensor"] == "SPOT6"
        assert "S6_SENSOR" in caplog.text
        assert len(d["att_df"]) == 8


# Synthetic DIMAP v1 fixtures (#179): a SPOT 5 across-track pair and an ALOS
# PRISM forward/backward pair, both written from ASP's reader spec
# (SPOT_XML.cc, PRISM_XML.cc) since no real delivery is available. Regenerate
# with tests/test_data/dimap_v1_synthetic/make_fixtures.py.
DIMAP_V1_DIR = TEST_DATA_DIR / "dimap_v1_synthetic"
SPOT5_DIR = DIMAP_V1_DIR / "spot5"
PRISM_DIR = DIMAP_V1_DIR / "prism"
SPOT5_EAST = SPOT5_DIR / "SPOT5_HRG1_SCENE_1A_east_synthetic.XML"
PRISM_FORWARD = PRISM_DIR / "PRISM_ALOS_forward_synthetic.XML"


class TestSpot5Metadata:
    """SPOT 5 DIMAP v1 reader (#179), mirroring ASP's SPOT_XML.cc."""

    @pytest.fixture
    def scene(self):
        return Spot5Metadata(image_list=[str(SPOT5_EAST)]).get_scene_dicts()[0]

    def test_detection(self):
        assert Spot5Metadata._is_camera_file(str(SPOT5_EAST)) is True
        assert Spot5Metadata.detect(str(SPOT5_DIR)) is True
        assert isinstance(sensor_for_directory(str(SPOT5_DIR)), Spot5Metadata)

    def test_other_sensors_do_not_claim_it(self):
        # DIMAP v1 shares the Dimap_Document root tag with the v2 products the
        # Pléiades reader handles; the Metadata_Id header is what separates
        # them, and neither reader may claim the other's files.
        assert PleiadesMetadata._is_camera_file(str(SPOT5_EAST)) is False
        assert PrismMetadata._is_camera_file(str(SPOT5_EAST)) is False
        assert WorldViewMetadata._is_camera_file(str(SPOT5_EAST)) is False

    def test_scene_identity(self, scene):
        assert scene["sensor"] == "SPOT5"
        # DATASET_NAME is a human-readable string; scene ids reach figure
        # filenames, so whitespace is collapsed.
        assert scene["catid"] == "SPOT5_HRG1_SCENE_1A_EAST_SYNTHETIC"
        assert " " not in scene["catid"]
        # The scene center time (what ASP reads), not IMAGING_TIME.
        assert scene["date"].strftime("%Y-%m-%dT%H:%M:%S") == "2008-03-04T12:31:03"
        assert scene["geom"].is_valid
        assert len(scene["geom"].exterior.coords) == 5  # 4 corners, closed

    def test_summary_fields_from_scene_source(self, scene):
        assert scene["meansatel"] == pytest.approx(90 - 15.021, abs=0.01)
        assert scene["meanoffnadirviewangle"] == pytest.approx(13.427)
        assert scene["meanproductgsd"] == pytest.approx(2.5)
        assert scene["meansunaz"] == pytest.approx(151.882)
        assert scene["meansunel"] == pytest.approx(42.316)
        # DIMAP v1 reports no satellite azimuth and no cloud cover: those
        # degrade to NaN rather than being omitted or guessed (#163).
        assert np.isnan(scene["meansataz"])
        assert np.isnan(scene["cloudcover"])
        assert scene["scandir"] is None and scene["tdi"] is None

    def test_eph_gdf(self, scene):
        eph_gdf = scene["eph_gdf"]
        assert isinstance(eph_gdf.index, pd.DatetimeIndex)
        assert len(eph_gdf) == 5
        radius = np.sqrt(eph_gdf["x"] ** 2 + eph_gdf["y"] ** 2 + eph_gdf["z"] ** 2)
        assert ((radius > 7.15e6) & (radius < 7.25e6)).all()  # ~822 km altitude
        for n in ["11", "12", "13", "22", "23", "33"]:
            assert eph_gdf[f"cov_{n}"].isna().all()

    def test_att_df_is_roll_pitch_yaw_in_degrees(self, scene):
        att_df = scene["att_df"]
        assert isinstance(att_df.index, pd.DatetimeIndex)
        assert list(att_df.columns[:3]) == ["roll", "pitch", "yaw"]
        assert "q1" not in att_df.columns
        assert att_df.attrs["rpy_frame"] == "SPOT Geometry Handbook navigation frame"
        for n in ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]:
            assert att_df[f"cov_{n}"].isna().all()

    def test_att_df_pinned_to_raw_fixture_values(self):
        # Pin both the unit conversion (the file stores radians; ASP feeds the
        # values straight to sin/cos) and the tag-to-column mapping: the file
        # lists YAW, PITCH, ROLL in that order, so a positional read would
        # silently swap roll and yaw.
        import math
        import xml.etree.ElementTree as ET

        first = (
            ET.parse(SPOT5_EAST)
            .getroot()
            .find(".//Corrected_Attitudes/Corrected_Attitude/Angles")
        )
        row = (
            Spot5Metadata(image_list=[str(SPOT5_EAST)])
            .get_scene_dicts()[0]["att_df"]
            .iloc[0]
        )
        for column, tag in (("roll", "ROLL"), ("pitch", "PITCH"), ("yaw", "YAW")):
            assert row[column] == pytest.approx(
                math.degrees(float(first.findtext(tag))), rel=1e-12
            )
        # Sanity check on the fixture itself: an off-nadir acquisition.
        assert row["roll"] == pytest.approx(15.0, abs=0.01)

    def test_center_element_is_not_a_corner(self, tmp_path):
        # Some DIMAP v1 products add a scene-center block with the same
        # FRAME_LON/FRAME_LAT children as the corners. Selecting by the
        # Vertex tag skips it, so the footprint stays a quadrilateral.
        text = SPOT5_EAST.read_text().replace(
            "    <SCENE_ORIENTATION>",
            "    <Center>\n"
            "      <FRAME_LON>26.816671</FRAME_LON>\n"
            "      <FRAME_LAT>43.098818</FRAME_LAT>\n"
            "      <FRAME_ROW>6000</FRAME_ROW>\n"
            "      <FRAME_COL>6000</FRAME_COL>\n"
            "    </Center>\n"
            "    <SCENE_ORIENTATION>",
        )
        f = tmp_path / "with_center.XML"
        f.write_text(text)
        geom = Spot5Metadata(image_list=[str(f)]).get_scene_dicts()[0]["geom"]
        assert len(geom.exterior.coords) == 5  # 4 corners, closed

    def test_unexpected_corner_count_warns(self, tmp_path, caplog):
        # Neither reader is validated against a real delivery, so a product
        # that yields something other than four corners must say so rather
        # than silently produce a differently-shaped footprint.
        text = SPOT5_EAST.read_text().replace(
            "    <SCENE_ORIENTATION>",
            "    <Vertex>\n"
            "      <FRAME_LON>26.9</FRAME_LON>\n"
            "      <FRAME_LAT>43.2</FRAME_LAT>\n"
            "    </Vertex>\n"
            "    <SCENE_ORIENTATION>",
        )
        f = tmp_path / "five_corners.XML"
        f.write_text(text)
        with caplog.at_level("WARNING"):
            Spot5Metadata(image_list=[str(f)]).get_scene_dicts()
        assert "5 corner vertices, expected 4" in caplog.text

    def test_missing_attitude_raises(self, tmp_path):
        text = SPOT5_EAST.read_text()
        text = re.sub(r"<Angles>.*?</Angles>", "", text, flags=re.DOTALL)
        f = tmp_path / "no_attitude.XML"
        f.write_text(text)
        with pytest.raises(ValueError, match="No attitude found"):
            Spot5Metadata(image_list=[str(f)]).get_scene_dicts()

    def test_other_spot_missions_are_skipped(self, tmp_path, caplog):
        # Only SPOT 5 is an ASP stereo session; a SPOT 4 scene is rejected
        # with a warning naming it rather than parsed as if it were SPOT 5.
        text = SPOT5_EAST.read_text().replace(
            "<MISSION_INDEX>5</MISSION_INDEX>", "<MISSION_INDEX>4</MISSION_INDEX>"
        )
        f = tmp_path / "SPOT4.XML"
        f.write_text(text)
        with caplog.at_level("WARNING"):
            assert Spot5Metadata._is_camera_file(str(f)) is False
        assert "SPOT 4 scene skipped" in caplog.text

    def test_pair_geometry(self):
        from asp_plot.stereopair_metadata_parser import StereopairMetadataParser

        p = StereopairMetadataParser(directory=str(SPOT5_DIR)).get_pair_dict()
        assert p["intersection_area"] > 100
        # Without a satellite azimuth there is no convergence angle to
        # compute; it degrades to NaN instead of raising (#163).
        assert np.isnan(p["conv_ang"])


class TestPrismMetadata:
    """ALOS PRISM reader (#179), mirroring ASP's PRISM_XML.cc."""

    @pytest.fixture
    def scene(self):
        return PrismMetadata(image_list=[str(PRISM_FORWARD)]).get_scene_dicts()[0]

    def test_detection(self):
        assert PrismMetadata._is_camera_file(str(PRISM_FORWARD)) is True
        assert isinstance(sensor_for_directory(str(PRISM_DIR)), PrismMetadata)
        # Gated on METADATA_PROFILE == "ALOS", exactly as ASP is.
        assert Spot5Metadata._is_camera_file(str(PRISM_FORWARD)) is False
        assert PleiadesMetadata._is_camera_file(str(PRISM_FORWARD)) is False

    def test_scene_identity(self, scene):
        # INSTRUMENT carries the view, which is what distinguishes the scenes
        # of a triplet.
        assert scene["sensor"] == "ALOS-PRISM_FORWARD"
        assert scene["catid"] == "ALPSMF_SYNTHETIC_FORWARD"
        assert scene["date"].strftime("%Y-%m-%dT%H:%M:%S") == "2007-05-19T01:23:45"
        assert scene["geom"].is_valid

    def test_att_df_degrees_as_delivered(self, scene):
        att_df = scene["att_df"]
        assert list(att_df.columns[:3]) == ["roll", "pitch", "yaw"]
        # PRISM angles are already in degrees (ASP's rollPitchYaw converts
        # them to radians), so they pass through unchanged.
        assert att_df["roll"].iloc[0] == pytest.approx(0.0521)
        assert att_df["pitch"].iloc[0] == pytest.approx(-0.0312)
        assert att_df["yaw"].iloc[0] == pytest.approx(0.1187)
        # Same orbital frame and Euler convention as the quaternion sensors'
        # computed angles, so the label says so.
        assert att_df.attrs["rpy_frame"] == "orbital frame (along, across, down)"

    def test_eph_gdf(self, scene):
        eph_gdf = scene["eph_gdf"]
        assert len(eph_gdf) == 6
        radius = np.sqrt(eph_gdf["x"] ** 2 + eph_gdf["y"] ** 2 + eph_gdf["z"] ** 2)
        assert ((radius > 7.02e6) & (radius < 7.12e6)).all()  # ~692 km altitude
        assert eph_gdf.crs.to_epsg() == 4978

    def test_summary_fields_degrade(self, scene):
        assert scene["meansunaz"] == pytest.approx(134.712)
        for key in [
            "meansataz",
            "meansatel",
            "meanoffnadirviewangle",
            "meanproductgsd",
            "cloudcover",
        ]:
            assert np.isnan(scene[key])

    def test_missing_ephemeris_raises(self, tmp_path):
        text = re.sub(
            r"<Ephemeris>.*?</Ephemeris>",
            "",
            PRISM_FORWARD.read_text(),
            flags=re.DOTALL,
        )
        f = tmp_path / "no_ephemeris.XML"
        f.write_text(text)
        with pytest.raises(ValueError, match="No ephemeris found"):
            PrismMetadata(image_list=[str(f)]).get_scene_dicts()


class TestDimapV1SpecOnlyWarning:
    """Both DIMAP v1 readers warn once that they are unvalidated (#179)."""

    @pytest.fixture(autouse=True)
    def _reset_warned(self, monkeypatch):
        monkeypatch.setattr(dimap_v1_module, "_warned_spec_only", set())

    def test_warns_once_per_reader(self, caplog):
        reader = Spot5Metadata(image_list=[str(SPOT5_EAST)])
        with caplog.at_level("WARNING"):
            reader.get_scene_dicts()
        assert "SPOT5 metadata support" in caplog.text
        assert "issues/179" in caplog.text
        caplog.clear()
        with caplog.at_level("WARNING"):
            reader.get_scene_dicts()
        assert "ASP reader spec" not in caplog.text

    def test_each_reader_warns_for_itself(self, caplog):
        Spot5Metadata(image_list=[str(SPOT5_EAST)]).get_scene_dicts()
        caplog.clear()
        with caplog.at_level("WARNING"):
            PrismMetadata(image_list=[str(PRISM_FORWARD)]).get_scene_dicts()
        assert "PRISM metadata support" in caplog.text


class TestWorldViewSceneDictDegradation:
    """#163: optional tags degrade to "not provided" instead of crashing."""

    @staticmethod
    def _strip_tags(src, dst, tags):
        text = Path(src).read_text()
        for tag in tags:
            text = re.sub(rf"<{tag}>[^<]*</{tag}>", "", text)
        Path(dst).write_text(text)

    @pytest.fixture
    def stripped_reader(self, tmp_path):
        # dg_mosaic can strip image tags; Multi products carry per-band TDI.
        self._strip_tags(
            CAM_A,
            tmp_path / "stripped.xml",
            ["TDILEVEL", "SCANDIRECTION", "CLOUDCOVER", "MEANSUNAZ"],
        )
        shutil.copy(CAM_B, tmp_path / "b.xml")
        return WorldViewMetadata(directory=str(tmp_path))

    def test_missing_optional_tags_degrade(self, stripped_reader):
        d = next(
            d
            for d in stripped_reader.get_scene_dicts()
            if d["catid"] == "10300100D0772D00"
        )
        assert d["tdi"] is None
        assert d["scandir"] is None
        assert np.isnan(d["cloudcover"])
        assert np.isnan(d["meansunaz"])
        # Untouched fields still parse.
        assert not np.isnan(d["meansataz"])
        assert d["sensor"] == "WV02"

    def test_identity_core_still_strict(self, tmp_path):
        # A camera XML without the identity core (here: FIRSTLINETIME) is an
        # error, not a degraded scene.
        self._strip_tags(CAM_A, tmp_path / "no_date.xml", ["FIRSTLINETIME"])
        shutil.copy(CAM_B, tmp_path / "b.xml")
        reader = WorldViewMetadata(directory=str(tmp_path))
        with pytest.raises(ValueError, match="FIRSTLINETIME"):
            reader.get_scene_dicts()
