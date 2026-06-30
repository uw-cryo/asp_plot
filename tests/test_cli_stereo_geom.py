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
