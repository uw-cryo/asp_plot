"""Tests for reconstructing mapproject commands from GeoTIFF metadata (#96)."""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from asp_plot.mapproject import find_mapproject_commands, reconstruct_mapproject_command

# A realistic ASP mapproject metadata signature, mirroring what gdalinfo shows
# on a WorldView _corr_map.tif (RPC session, bundle-adjusted).
ASP_TAGS = {
    "INPUT_IMAGE_FILE": "scene_corr.tif",
    "CAMERA_FILE": "scene.xml",
    "DEM_FILE": "ref/cop30.tif",
    "CAMERA_MODEL_TYPE": "rpc",
    "BUNDLE_ADJUST_PREFIX": "ba/run",
}


def _write_tagged_tif(path, tags, crs="EPSG:32616", origin=(735685.0, 3727694.0)):
    """Write a tiny GeoTIFF with the given ASP metadata tags."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs=CRS.from_user_input(crs) if crs else None,
        transform=from_origin(origin[0], origin[1], 0.5, 0.5),
    ) as dst:
        dst.write(np.ones((1, 4, 4), dtype="float32"))
        if tags:
            dst.update_tags(**tags)


class TestReconstruct:
    def test_full_command(self, tmp_path):
        fn = tmp_path / "scene_corr_map.tif"
        _write_tagged_tif(str(fn), ASP_TAGS)
        cmd = reconstruct_mapproject_command(str(fn))

        # Resolved session, srs, grid, projwin, BA prefix, then positional args.
        assert cmd.startswith("mapproject -t rpc --t_srs EPSG:32616 --tr 0.5 ")
        assert "--t_projwin 735685 3727692 735687 3727694" in cmd
        assert "--bundle-adjust-prefix ba/run" in cmd
        # Positional order: DEM, input image, camera, output (basename).
        assert cmd.endswith("ref/cop30.tif scene_corr.tif scene.xml scene_corr_map.tif")

    def test_no_bundle_adjust_prefix_omitted(self, tmp_path):
        tags = dict(ASP_TAGS, BUNDLE_ADJUST_PREFIX="NONE")
        fn = tmp_path / "scene_map.tif"
        _write_tagged_tif(str(fn), tags)
        cmd = reconstruct_mapproject_command(str(fn))
        assert "--bundle-adjust-prefix" not in cmd

    def test_non_epsg_crs_falls_back_to_proj4(self, tmp_path):
        # A custom stereographic frame with no EPSG code (cf. jitter solving).
        proj = (
            "+proj=stere +lat_0=46.85 +lon_0=-121.76 +k=1 +x_0=0 +y_0=0 "
            "+datum=WGS84 +units=m +no_defs"
        )
        tags = dict(ASP_TAGS, CAMERA_MODEL_TYPE="csm")
        fn = tmp_path / "out.map.tif"
        _write_tagged_tif(str(fn), tags, crs=proj, origin=(-45315.0, 56025.0))
        cmd = reconstruct_mapproject_command(str(fn))
        assert '--t_srs "+proj=stere' in cmd
        assert "-t csm" in cmd

    def test_projwin_not_scientific_notation(self, tmp_path):
        # Large UTM northings must not render as 3.7e+06 (unusable on the CLI).
        import re

        fn = tmp_path / "scene_map.tif"
        _write_tagged_tif(str(fn), ASP_TAGS)
        cmd = reconstruct_mapproject_command(str(fn))
        # No <digit>e[+-]<digit> exponent tokens (would not match "bundle-adjust").
        assert not re.search(r"\de[+-]\d", cmd)

    def test_non_mapproject_tif_returns_none(self, tmp_path):
        # A georeferenced raster with no ASP mapproject tags is not a candidate.
        fn = tmp_path / "plain.tif"
        _write_tagged_tif(str(fn), tags={})
        assert reconstruct_mapproject_command(str(fn)) is None

    def test_partial_tags_returns_none(self, tmp_path):
        # Missing the camera file -> not enough to reconstruct.
        fn = tmp_path / "partial.tif"
        _write_tagged_tif(str(fn), {"INPUT_IMAGE_FILE": "scene.tif"})
        assert reconstruct_mapproject_command(str(fn)) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert reconstruct_mapproject_command(str(tmp_path / "nope.tif")) is None


class TestFindCommands:
    def test_finds_pair_and_dedupes(self, tmp_path):
        # Left/right scenes in the root; one also reachable via a stereo subdir.
        _write_tagged_tif(
            str(tmp_path / "left_map.tif"),
            dict(ASP_TAGS, INPUT_IMAGE_FILE="left_corr.tif"),
        )
        _write_tagged_tif(
            str(tmp_path / "right_map.tif"),
            dict(ASP_TAGS, INPUT_IMAGE_FILE="right_corr.tif"),
        )
        # A non-mapproject tif should be ignored.
        _write_tagged_tif(str(tmp_path / "dem.tif"), tags={})

        cmds = find_mapproject_commands([str(tmp_path), None, "nonexistent"])
        assert len(cmds) == 2
        assert any("left_corr.tif" in c for c in cmds)
        assert any("right_corr.tif" in c for c in cmds)
        # Sorted for stable report output.
        assert cmds == sorted(cmds)

    def test_empty_when_no_mapproject_outputs(self, tmp_path):
        _write_tagged_tif(str(tmp_path / "run-DEM.tif"), tags={})
        assert find_mapproject_commands([str(tmp_path)]) == []

    def test_none_and_missing_dirs_skipped(self):
        assert find_mapproject_commands([None, "/no/such/dir"]) == []


@pytest.mark.parametrize("glob_name", ["s_map.tif", "s_proj.tif", "s.map.tif"])
def test_filename_conventions_discovered(tmp_path, glob_name):
    _write_tagged_tif(str(tmp_path / glob_name), ASP_TAGS)
    assert len(find_mapproject_commands([str(tmp_path)])) == 1
