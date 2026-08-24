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


class TestPcAlignReport:
    """pc_align_report() keys off literal substrings of pc_align's log; these
    pin the parse against real logs from before (2024-11) and after (ASP
    3.8.0-alpha) the 3.7.0 change that added the one-line error-stats summary
    (issue #146), so an upstream log-format change fails loudly here.
    """

    @pytest.fixture
    def alignment(self):
        return Alignment(directory="tests/test_data", dem_fn=DEM)

    def test_missing_log_returns_none(self, alignment):
        assert alignment.pc_align_report(output_prefix="pc_align/nope") is None

    def test_pre_370_log_has_percentiles_and_translation_only(self, alignment):
        report = alignment.pc_align_report(output_prefix="pc_align/pc_align_ground")
        assert report == pytest.approx(
            {
                "p16_beg": 0.379974,
                "p50_beg": 0.613481,
                "p84_beg": 0.823963,
                "p16_end": 0.0449552,
                "p50_end": 0.151377,
                "p84_end": 0.332819,
                "north_shift": report["north_shift"],
                "east_shift": report["east_shift"],
                "down_shift": report["down_shift"],
                "translation_magnitude": report["translation_magnitude"],
            }
        )
        # No 3.7.0-style stats line in this log -> no such keys, not NaN/None
        assert not {
            k
            for k in report
            if k.split("_")[0] in ("mean", "stddev", "rmse", "median", "nmad")
        }

    def test_380_log_parses_new_error_stats(self, alignment):
        report = alignment.pc_align_report(output_prefix="pc_align/pc_align_lola")
        # The pre-3.7.0 lines are unchanged in 3.8.0 and still parse
        assert report["p16_beg"] == pytest.approx(2.92937)
        assert report["p50_beg"] == pytest.approx(5.71217)
        assert report["p84_beg"] == pytest.approx(235.916)
        assert report["p50_end"] == pytest.approx(2.01482)
        assert report["north_shift"] == pytest.approx(-0.089497083)
        assert report["east_shift"] == pytest.approx(-1.1836275)
        assert report["down_shift"] == pytest.approx(4.7552564)
        assert report["translation_magnitude"] == pytest.approx(4.901168)
        # The new "Input/Output stats (meters):" lines
        assert report["mean_beg"] == pytest.approx(71.4632)
        assert report["stddev_beg"] == pytest.approx(134.758)
        assert report["rmse_beg"] == pytest.approx(152.535)
        assert report["median_beg"] == pytest.approx(5.71217)
        assert report["nmad_beg"] == pytest.approx(3.20963)
        assert report["mean_end"] == pytest.approx(69.3561)
        assert report["stddev_end"] == pytest.approx(136.126)
        assert report["rmse_end"] == pytest.approx(152.776)
        assert report["median_end"] == pytest.approx(2.01482)
        assert report["nmad_end"] == pytest.approx(2.07062)
        # Median is p50 under another name
        assert report["median_beg"] == report["p50_beg"]
        assert report["median_end"] == report["p50_end"]

    def test_380_log_key_order_matches_log_order(self, alignment):
        """The dict order becomes the alignment_report_df column order and
        hence the report-page table order."""
        report = alignment.pc_align_report(output_prefix="pc_align/pc_align_lola")
        assert list(report) == [
            "p16_beg",
            "p50_beg",
            "p84_beg",
            "mean_beg",
            "stddev_beg",
            "rmse_beg",
            "median_beg",
            "nmad_beg",
            "p16_end",
            "p50_end",
            "p84_end",
            "mean_end",
            "stddev_end",
            "rmse_end",
            "median_end",
            "nmad_end",
            "north_shift",
            "east_shift",
            "down_shift",
            "translation_magnitude",
        ]


class TestParsePcAlignStatsLine:
    def test_parses_all_five(self):
        from asp_plot.alignment import _parse_pc_align_stats_line

        line = (
            "2026-08-18 18:07:29 {0} [ console ] : Input stats (meters): "
            "Mean: 71.4632, StdDev: 134.758, RMSE: 152.535, Median: 5.71217, NMAD: 3.20963"
        )
        assert _parse_pc_align_stats_line(line, "beg") == pytest.approx(
            {
                "mean_beg": 71.4632,
                "stddev_beg": 134.758,
                "rmse_beg": 152.535,
                "median_beg": 5.71217,
                "nmad_beg": 3.20963,
            }
        )

    def test_integer_and_scientific_values(self):
        from asp_plot.alignment import _parse_pc_align_stats_line

        line = "Output stats (meters): Mean: 0, StdDev: 1e-05, RMSE: 2.5E+02, Median: 3, NMAD: -0.5"
        assert _parse_pc_align_stats_line(line, "end") == pytest.approx(
            {
                "mean_end": 0.0,
                "stddev_end": 1e-05,
                "rmse_end": 250.0,
                "median_end": 3.0,
                "nmad_end": -0.5,
            }
        )

    def test_missing_label_is_skipped_not_raised(self):
        from asp_plot.alignment import _parse_pc_align_stats_line

        assert _parse_pc_align_stats_line("Input stats (meters): Mean: 1.5", "beg") == {
            "mean_beg": 1.5
        }
