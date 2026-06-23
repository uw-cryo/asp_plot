import os

import pytest

from asp_plot.alignment import Alignment

DEM = "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"


class TestRunPcAlign:
    """The two public pc_align wrappers both delegate to _run_pc_align; the
    argv they build must match the pre-refactor inline commands byte for byte.
    """

    @pytest.fixture
    def captured(self, monkeypatch):
        calls = {}
        monkeypatch.setattr(
            "asp_plot.alignment.run_subprocess_command",
            lambda cmd: calls.__setitem__("cmd", cmd),
        )
        return calls

    @pytest.fixture
    def alignment(self, tmp_path):
        return Alignment(directory=str(tmp_path), dem_fn=DEM)

    def test_atl06sr_argv(self, alignment, captured, tmp_path):
        csv = tmp_path / "atl.csv"
        csv.write_text("x")
        alignment.pc_align_dem_to_atl06sr(atl06sr_csv=str(csv))
        assert captured["cmd"] == [
            "pc_align",
            "--max-displacement",
            "20",
            "--max-num-source-points",
            "10000000",
            "--alignment-method",
            "point-to-point",
            "--csv-format",
            "1:lon 2:lat 3:height_above_datum",
            "--compute-translation-only",
            "--output-prefix",
            os.path.join(str(tmp_path), "pc_align/pc_align"),
            DEM,
            str(csv),
        ]
        # The Earth/ICESat-2 path must not emit a --datum flag.
        assert "--datum" not in captured["cmd"]

    @pytest.mark.parametrize(
        "body, datum",
        [("moon", "D_MOON"), ("mars", "D_MARS")],
    )
    def test_planetary_argv(self, alignment, captured, tmp_path, body, datum):
        csv = tmp_path / "planet.csv"
        csv.write_text("x")
        alignment.pc_align_dem_to_planetary_csv(planetary_csv=str(csv), body=body)
        assert captured["cmd"] == [
            "pc_align",
            "--max-displacement",
            "500",
            "--max-num-source-points",
            "10000000",
            "--alignment-method",
            "point-to-point",
            "--csv-format",
            "1:lon 2:lat 3:radius_m",
            "--datum",
            datum,
            "--compute-translation-only",
            "--output-prefix",
            os.path.join(str(tmp_path), "pc_align/pc_align"),
            DEM,
            str(csv),
        ]

    def test_atl06sr_missing_csv_raises(self, alignment):
        with pytest.raises(ValueError, match="not found"):
            alignment.pc_align_dem_to_atl06sr(atl06sr_csv=None)

    def test_planetary_missing_csv_raises(self, alignment, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            alignment.pc_align_dem_to_planetary_csv(
                planetary_csv=str(tmp_path / "missing.csv"), body="mars"
            )

    def test_planetary_rejects_earth(self, alignment, captured, tmp_path):
        csv = tmp_path / "planet.csv"
        csv.write_text("x")
        with pytest.raises(ValueError, match="Unsupported body"):
            alignment.pc_align_dem_to_planetary_csv(
                planetary_csv=str(csv), body="earth"
            )
        # Rejected before any pc_align invocation.
        assert "cmd" not in captured

    def test_run_pc_align_max_displacement_passthrough(
        self, alignment, captured, tmp_path
    ):
        csv = tmp_path / "atl.csv"
        csv.write_text("x")
        alignment.pc_align_dem_to_atl06sr(atl06sr_csv=str(csv), max_displacement=42)
        assert captured["cmd"][captured["cmd"].index("--max-displacement") + 1] == "42"
