import matplotlib
from click.testing import CliRunner

from asp_plot.cli.stereo_geom import main

matplotlib.use("Agg")

CAM_A = "tests/test_data/10300100D0772D00.r100.xml"
CAM_B = "tests/test_data/10300100D12D7400.r100.xml"


def _run(args):
    # add_basemap False keeps the CLI offline (no contextily tile fetch).
    return CliRunner().invoke(main, args + ["--add_basemap", "False"])


class TestStereoGeomCli:
    def test_positional_xml_files(self, tmp_path):
        # The geom_plot *.XML case: shell hands the CLI a list of XML files.
        out = tmp_path / "from_files.png"
        result = _run(
            [
                CAM_A,
                CAM_B,
                "--output_directory",
                str(tmp_path),
                "--output_filename",
                out.name,
            ]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_glob_pattern(self, tmp_path):
        out = tmp_path / "from_glob.png"
        result = _run(
            [
                "tests/test_data/*.r100.xml",
                "--output_directory",
                str(tmp_path),
                "--output_filename",
                out.name,
            ]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_directory_flag_still_works(self, tmp_path):
        # Backward compatibility with the original --directory interface.
        out = tmp_path / "from_flag.png"
        result = _run(
            [
                "--directory",
                "tests/test_data",
                "--output_directory",
                str(tmp_path),
                "--output_filename",
                out.name,
            ]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_default_output_filename_from_directory(self, tmp_path):
        # With no --output_filename, the name derives from the base directory.
        result = _run([CAM_A, CAM_B, "--output_directory", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "test_data_stereo_geom.png").exists()

    def test_no_xml_inputs_errors(self, tmp_path):
        (tmp_path / "note.txt").write_text("not xml")
        result = _run(["--directory", str(tmp_path)])
        assert result.exit_code != 0

    def test_three_scenes_emit_overview_and_pairs(self, tmp_path):
        # More than two scenes -> overview + one figure per pair, derived from
        # the --output_filename stem.
        third = "tests/test_data/tiled_xmls/10200100A1865800.r100.xml"
        result = _run(
            [
                CAM_A,
                CAM_B,
                third,
                "--output_directory",
                str(tmp_path),
                "--output_filename",
                "geom.png",
            ]
        )
        assert result.exit_code == 0, result.output
        pngs = sorted(p.name for p in tmp_path.glob("*.png"))
        assert "geom_overview.png" in pngs
        # 1 overview + 3 pairs (3-choose-2).
        assert len(pngs) == 4
