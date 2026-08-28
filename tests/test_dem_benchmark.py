import os
import shutil

import matplotlib
import numpy as np
import pandas as pd
import pytest

from asp_plot.dem_benchmark import (
    STATS_COLUMNS,
    DEMBenchmark,
    intersection_error_path,
    label_from_dem_path,
    parse_dem_specs,
)

matplotlib.use("Agg")

# Two tiny Utqiagvik DEMs in one UTM zone that overlap each other and the
# ICESat-2 fixture (153k ATL06-SR points with the WorldCover column cached).
DEM_REF = "tests/test_data/ref_dem.tif"
DEM_STEREO = "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"
# A DEM in another UTM zone, far from the points: nothing overlaps it.
DEM_FAR = "tests/test_data/no_mapproj/stereo/run-DEM.tif"
PARQUET = "tests/test_data/icesat_data/atl06sr_all.parquet"
PC_ALIGN_FIXTURES = "tests/test_data/pc_align"


class TestLabels:
    def test_generic_asp_prefix_uses_folder(self):
        assert (
            label_from_dem_path("atlanta_mvs/stereo_mvs3/run-DEM.tif") == "stereo_mvs3"
        )

    def test_named_dem_uses_stem(self):
        assert (
            label_from_dem_path("atlanta_mvs/pairwise_mosaic-DEM.tif")
            == "pairwise_mosaic"
        )
        assert label_from_dem_path("tests/test_data/ref_dem.tif") == "ref_dem"

    def test_parse_specs_labels_order_and_duplicates(self):
        dems = parse_dem_specs(
            [
                "MVS=stereo_mvs3/run-DEM.tif",
                "a/run-DEM.tif",
                "b/run-DEM.tif",
                "x/run-DEM.tif",
                "y/run-DEM.tif",
            ]
        )
        assert list(dems) == ["MVS", "a", "b", "x", "y"]
        assert dems["MVS"] == "stereo_mvs3/run-DEM.tif"
        # Two run-DEM.tif in same-named folders must not collapse into one row.
        dup = parse_dem_specs(["one/run/run-DEM.tif", "two/run/run-DEM.tif"])
        assert len(dup) == 2

    def test_intersection_error_sibling(self):
        assert intersection_error_path(DEM_STEREO) == DEM_STEREO.replace(
            "-DEM.tif", "-IntersectionErr.tif"
        )
        assert intersection_error_path(DEM_REF) is None


class TestValidation:
    def test_missing_dem_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            DEMBenchmark(str(tmp_path), {"x": "nope-DEM.tif"}, parquet=PARQUET)

    def test_earth_requires_parquet(self, tmp_path):
        with pytest.raises(ValueError, match="parquet"):
            DEMBenchmark(str(tmp_path), [DEM_STEREO])

    def test_unknown_reference_raises(self, tmp_path):
        with pytest.raises(ValueError, match="reference"):
            DEMBenchmark(str(tmp_path), [DEM_STEREO], parquet=PARQUET, reference="zzz")

    def test_parquet_string_becomes_key_mapping(self, tmp_path):
        bench = DEMBenchmark(str(tmp_path), [DEM_STEREO], parquet=PARQUET)
        assert bench.parquet == {"all": PARQUET}
        assert bench.body == "earth"
        assert bench.aoi_bounds is not None

    def test_stats_required_before_plotting(self, tmp_path):
        bench = DEMBenchmark(str(tmp_path), [DEM_STEREO], parquet=PARQUET)
        with pytest.raises(ValueError, match="run\\(\\)"):
            bench.summary_plot()


@pytest.fixture
def bench(tmp_path):
    return DEMBenchmark(
        str(tmp_path),
        {"ref": DEM_REF, "stereo": DEM_STEREO},
        parquet=PARQUET,
        reference="ref",
        title="test",
    )


@pytest.fixture
def fake_pc_align(monkeypatch):
    """Stand in for the pc_align binary: drop a real log + transform at the
    requested output prefix (the Utqiagvik fixture, whose centroid sits on the
    test DEMs) so pc_align_report / apply_dem_translation run for real."""
    calls = []

    def fake(command):
        calls.append(list(command))
        prefix = command[command.index("--output-prefix") + 1]
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        shutil.copy(
            os.path.join(PC_ALIGN_FIXTURES, "pc_align_ground-log-pc_align.txt"),
            f"{prefix}-log-pc_align-01-01-0000-1.txt",
        )
        shutil.copy(
            os.path.join(PC_ALIGN_FIXTURES, "pc_align_ground-transform.txt"),
            f"{prefix}-transform.txt",
        )
        return 0

    monkeypatch.setattr("asp_plot.alignment.run_subprocess_command", fake)
    return calls


class TestRun:
    def test_without_pc_align(self, bench):
        df = bench.run(pc_align=False)
        assert list(df.columns) == STATS_COLUMNS
        assert list(df["label"]) == ["ref", "stereo"]
        assert (df["n_points"] > 0).all()
        assert np.isfinite(df[["dh_median_m", "dh_nmad_m", "dh_rmse_m"]]).all().all()
        assert df["dh_nmad_m"].gt(0).all()
        # Coverage inside the common footprint, area in km² from the GSD.
        assert df["valid_pct"].between(0, 100).all()
        assert (df["valid_area_km2"] > 0).all()
        assert bench.aoi_area_km2() > 0
        # Only the ASP-named DEM has an IntersectionErr sibling.
        assert np.isnan(df.set_index("label").loc["ref", "ie_median_m"])
        assert np.isfinite(df.set_index("label").loc["stereo", "ie_median_m"])
        # Alignment columns stay NaN when pc_align is off.
        assert df[["translation_m", "dh_aligned_nmad_m"]].isna().all().all()
        # Reference row is zero by definition; the other is a real difference.
        by = df.set_index("label")
        assert by.loc["ref", "vs_ref_median_m"] == 0.0
        assert np.isfinite(by.loc["stereo", "vs_ref_median_m"])
        # Per-DEM Altimetry objects are kept for follow-up figures.
        assert set(bench.altimetry) == {"ref", "stereo"}
        assert set(bench.dh) == {"ref", "stereo"}
        assert not bench.dh_aligned

    def test_with_pc_align(self, bench, fake_pc_align):
        df = bench.run(pc_align=True, minimum_points=10)
        assert len(fake_pc_align) == 2
        by = df.set_index("label")
        for label in ("ref", "stereo"):
            assert np.isfinite(by.loc[label, "translation_m"])
            assert np.isfinite(by.loc[label, "dh_aligned_median_m"])
            assert np.isfinite(by.loc[label, "dh_aligned_nmad_m"])
            label_dir = bench.label_directory(label)
            # Products land in the benchmark folder, not next to the DEM.
            assert os.path.exists(
                os.path.join(label_dir, "pc_align", "pc_align_all-transform.txt")
            )
            translated = [
                f
                for f in os.listdir(label_dir)
                if f.endswith("_pc_align_translated.tif")
            ]
            assert len(translated) == 1
            assert label in bench.dh_aligned
        assert not os.path.exists(
            DEM_STEREO.replace(".tif", "_pc_align_translated.tif")
        )
        # The fixture log reports a 0.859 m translation magnitude.
        assert by.loc["stereo", "translation_m"] == pytest.approx(0.859, abs=0.01)
        # Second run reuses the existing pc_align products: no new subprocess.
        bench.run(pc_align=True, minimum_points=10)
        assert len(fake_pc_align) == 2

    def test_minimum_points_skips_alignment(self, bench, fake_pc_align):
        df = bench.run(pc_align=True, minimum_points=10**9)
        assert not fake_pc_align
        assert df["translation_m"].isna().all()

    def test_missing_pc_align_binary_degrades(self, bench, monkeypatch):
        def missing(command):
            raise FileNotFoundError("pc_align")

        monkeypatch.setattr("asp_plot.alignment.run_subprocess_command", missing)
        df = bench.run(pc_align=True, minimum_points=10)
        assert df["translation_m"].isna().all()
        assert np.isfinite(df["dh_nmad_m"]).all()
        assert bench._pc_align_available is False

    def test_dem_without_points_or_overlap(self, tmp_path):
        bench = DEMBenchmark(
            str(tmp_path),
            {"stereo": DEM_STEREO, "far": DEM_FAR},
            parquet=PARQUET,
            reference="stereo",
        )
        # Footprints in different UTM zones do not intersect: own extents.
        assert bench.aoi_bounds is None
        df = bench.run(pc_align=False).set_index("label")
        assert df.loc["far", "n_points"] == 0
        assert np.isnan(df.loc["far", "dh_median_m"])
        assert np.isnan(df.loc["far", "vs_ref_median_m"])
        assert df.loc["far", "valid_pct"] > 0
        assert df.loc["stereo", "n_points"] > 0

    def test_own_extent_aoi(self, tmp_path):
        bench = DEMBenchmark(str(tmp_path), [DEM_STEREO], parquet=PARQUET, aoi=None)
        assert bench.aoi_bounds is None
        assert bench.aoi_area_km2() is None
        df = bench.run(pc_align=False)
        assert df.loc[0, "valid_pct"] > 0

    def test_explicit_aoi_tuple(self, tmp_path):
        bench = DEMBenchmark(
            str(tmp_path),
            [DEM_STEREO],
            parquet=PARQUET,
            aoi=(575608, 7908286, 594558, 7923768),
        )
        assert bench.aoi_bounds == (575608, 7908286, 594558, 7923768)


class TestOutput:
    def test_save_stats_and_figures(self, bench, fake_pc_align, tmp_path):
        bench.run(pc_align=True, minimum_points=10)
        csv_fn = bench.save_stats(str(tmp_path / "out" / "stats.csv"))
        assert list(pd.read_csv(csv_fn).columns) == STATS_COLUMNS

        fig = bench.summary_plot(save_dir=str(tmp_path), fig_fn="summary.png")
        assert os.path.exists(tmp_path / "summary.png")
        # Best-first ordering by post-alignment NMAD, top row first.
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        expected = list(bench.stats_df.sort_values("dh_aligned_nmad_m")["label"])
        assert labels == expected
        # Coverage, IntersectionErr (the stereo DEM has one), median, NMAD.
        assert len(fig.axes) == 4
        matplotlib.pyplot.close(fig)

        fig = bench.histogram_plot(save_dir=str(tmp_path), fig_fn="hist.png")
        assert os.path.exists(tmp_path / "hist.png")
        assert "after pc_align" in fig.axes[0].get_xlabel()
        matplotlib.pyplot.close(fig)

    def test_plots_without_alignment(self, bench, tmp_path):
        bench.run(pc_align=False)
        fig = bench.summary_plot(sort=False)
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == ["ref", "stereo"]
        matplotlib.pyplot.close(fig)
        fig = bench.histogram_plot()
        assert "before pc_align" in fig.axes[0].get_xlabel()
        matplotlib.pyplot.close(fig)


class TestCLI:
    def test_cli_smoke(self, tmp_path):
        from click.testing import CliRunner

        from asp_plot.cli.dem_benchmark import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "reference=" + DEM_REF,
                DEM_STEREO,
                "--parquet",
                PARQUET,
                "--directory",
                str(tmp_path),
                "--no-pc-align",
                "--reference",
                "reference",
                "--output-filename",
                "bench.png",
            ],
        )
        assert result.exit_code == 0, result.output
        for fn in ("bench.png", "bench_histogram.png", "bench.csv"):
            assert os.path.exists(tmp_path / fn), fn
        df = pd.read_csv(tmp_path / "bench.csv")
        assert list(df["label"]) == ["reference", "date_time_left_right_1m"]
        assert "reference" in result.output
