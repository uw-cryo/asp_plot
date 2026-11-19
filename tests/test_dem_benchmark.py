import os
import shutil

import matplotlib
import numpy as np
import pandas as pd
import pytest

from asp_plot.dem_benchmark import (
    SLOPE_STATS_COLUMNS,
    STATS_COLUMNS,
    DEMBenchmark,
    intersection_error_path,
    label_from_dem_path,
    paired_block_bootstrap,
    parse_dem_specs,
    reference_dem_from_logs,
    slope_at_points,
)
from asp_plot.terrain import SLOPE_MIN_POINTS, slope_pixel_size, slope_source_label

matplotlib.use("Agg")

# Two tiny Utqiagvik DEMs in one UTM zone that overlap each other and the
# ICESat-2 fixture (153k ATL06-SR points with the WorldCover column cached).
DEM_REF = "tests/test_data/ref_dem.tif"
DEM_STEREO = "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"
# A DEM in another UTM zone, far from the points: nothing overlaps it.
DEM_FAR = "tests/test_data/no_mapproj/stereo/run-DEM.tif"
PARQUET = "tests/test_data/icesat_data/atl06sr_all.parquet"
PC_ALIGN_FIXTURES = "tests/test_data/pc_align"
# The reference DEM the stereo logs beside DEM_STEREO name.
LOGGED_REFERENCE_DEM = (
    "/nobackup/bpurint1/data/utqiagvik/COP/COP30_utqiagvik_lzw-adj_proj.tif"
)


@pytest.fixture(autouse=True)
def logged_reference_dems(monkeypatch):
    """
    Stand in for the reference DEMs read from stereo logs: {DEM path: path}.

    Empty unless a test fills it, so slope_dem="auto" finds nothing and the
    tests do not depend on whether LOGGED_REFERENCE_DEM exists on the machine.
    """
    found = {}
    monkeypatch.setattr(
        "asp_plot.dem_benchmark.reference_dem_from_logs", lambda fn: found.get(fn)
    )
    return found


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

    def test_unknown_filter_group_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown ESA WorldCover group"):
            DEMBenchmark(
                str(tmp_path),
                {"ref": DEM_REF},
                parquet=PARQUET,
                filter_out=["water", "tree"],
            )

    def test_stats_required_before_plotting(self, tmp_path):
        bench = DEMBenchmark(str(tmp_path), [DEM_STEREO], parquet=PARQUET)
        with pytest.raises(ValueError, match="run\\(\\)"):
            bench.summary_plot()


class TestPairedBlockBootstrap:
    def test_shapes_and_determinism(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(3, 400))
        blocks = np.repeat(np.arange(20), 20)
        a = paired_block_bootstrap(values, blocks, n_boot=50, seed=3)
        b = paired_block_bootstrap(values, blocks, n_boot=50, seed=3)
        assert a["median"].shape == a["nmad"].shape == (50, 3)
        assert a["n_blocks"] == 20
        np.testing.assert_array_equal(a["nmad"], b["nmad"])
        assert not np.array_equal(
            a["nmad"], paired_block_bootstrap(values, blocks, 50, seed=4)["nmad"]
        )

    def test_points_when_no_blocks(self):
        values = np.random.default_rng(0).normal(size=(2, 100))
        out = paired_block_bootstrap(values, None, n_boot=10)
        assert out["n_blocks"] == 100

    def test_pairing_cancels_shared_variation(self):
        # Two series that differ by a constant: every replicate's medians
        # differ by exactly that constant, and the NMADs are identical.
        rng = np.random.default_rng(2)
        base = rng.standard_t(3, size=500)
        values = np.vstack([base, base + 0.7])
        blocks = np.repeat(np.arange(25), 20)
        out = paired_block_bootstrap(values, blocks, n_boot=30)
        np.testing.assert_allclose(out["median"][:, 1] - out["median"][:, 0], 0.7)
        np.testing.assert_allclose(out["nmad"][:, 1], out["nmad"][:, 0])

    def test_blocks_widen_intervals_for_correlated_residuals(self):
        # A per-block offset makes points within a block correlated; block
        # resampling must spread the median more than point resampling.
        rng = np.random.default_rng(5)
        offsets = np.repeat(rng.normal(scale=2.0, size=30), 40)
        values = (offsets + rng.normal(scale=0.2, size=offsets.size))[None, :]
        blocks = np.repeat(np.arange(30), 40)
        by_block = paired_block_bootstrap(values, blocks, n_boot=200)["median"]
        by_point = paired_block_bootstrap(values, None, n_boot=200)["median"]
        assert by_block.std() > 3 * by_point.std()

    def test_nan_ignored(self):
        values = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]] * 2)
        out = paired_block_bootstrap(values, None, n_boot=5)
        assert np.isfinite(out["median"]).all()

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            paired_block_bootstrap(np.arange(5.0), None, n_boot=2)


@pytest.fixture
def bench(tmp_path):
    return DEMBenchmark(
        str(tmp_path),
        {"ref": DEM_REF, "stereo": DEM_STEREO},
        parquet=PARQUET,
        reference="ref",
        n_bootstrap=100,
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

    def test_shared_points_and_bootstrap(self, bench):
        df = bench.run(pc_align=False).set_index("label")
        # The shared set is the intersection of what each DEM kept, and every
        # DEM is scored on exactly those points.
        expected = bench.dh["ref"].index.intersection(bench.dh["stereo"].index)
        assert bench.shared_ids.equals(expected.sort_values())
        n_shared = len(bench.shared_ids)
        assert 0 < n_shared <= df["n_points"].min()
        assert (df["n_shared"] == n_shared).all()
        for label in ("ref", "stereo"):
            on_shared = bench.dh[label].reindex(bench.shared_ids)
            assert df.loc[label, "dh_shared_median_m"] == pytest.approx(
                np.median(on_shared)
            )
            assert df.loc[label, "dh_shared_nmad_m"] > 0
        # Own-point stats are still there and generally differ.
        assert np.isfinite(df["dh_nmad_m"]).all()
        assert df[["dh_aligned_shared_nmad_m", "median_ci_low_m"]].notna().any().any()
        # Points are identified consistently across DEMs: same block per id.
        assert bench.block_name == "beam track"
        assert bench.blocks.index.is_unique
        # Bootstrap: stage is pre-alignment, intervals bracket the estimate,
        # the best DEM has a zero difference to itself, P(first) sums to 1.
        assert bench.bootstrap["stage"] == "before"
        assert bench.bootstrap["block"] == "beam track"
        assert bench.bootstrap["n_replicates"] == 100
        assert bench.bootstrap["nmad"].shape == (100, 2)
        for label in ("ref", "stereo"):
            row = df.loc[label]
            assert (
                row["nmad_ci_low_m"] <= row["dh_shared_nmad_m"] <= row["nmad_ci_high_m"]
            )
            assert (
                row["median_ci_low_m"]
                <= row["dh_shared_median_m"]
                <= row["median_ci_high_m"]
            )
        best = bench.bootstrap["best"]
        assert best == df["dh_shared_nmad_m"].idxmin()
        assert df.loc[best, "nmad_vs_best_ci_low_m"] == 0.0
        assert df.loc[best, "nmad_vs_best_ci_high_m"] == 0.0
        assert df["p_best"].sum() == pytest.approx(1.0)
        assert df["p_best"].between(0, 1).all()
        # Blocks resampled are the beam tracks among the shared points.
        tracks = bench.blocks.reindex(bench.shared_ids).nunique()
        assert bench.bootstrap["n_blocks"] == tracks

    def test_filter_out_list(self, bench, tmp_path):
        # Dropping a second land-cover group leaves fewer points, for every
        # DEM alike, and the figure says which groups were dropped.
        default = bench.run(pc_align=False).set_index("label")
        assert "without" not in bench.summary_plot()._suptitle.get_text()

        both = DEMBenchmark(
            str(tmp_path),
            {"ref": DEM_REF, "stereo": DEM_STEREO},
            parquet=PARQUET,
            filter_out=["water", "low_vegetation"],
            n_bootstrap=0,
        )
        df = both.run(pc_align=False).set_index("label")
        assert (df["n_points"] < default["n_points"]).all()
        assert df["n_shared"].iloc[0] < default["n_shared"].iloc[0]
        classes = both.altimetry["ref"].atl06sr_processing_levels_filtered["all"][
            "esa_worldcover.value"
        ]
        assert not classes.isin([80, 90, 30, 100]).any()
        title = both.summary_plot()._suptitle.get_text()
        assert "without water, low vegetation" in title

        unfiltered = DEMBenchmark(
            str(tmp_path),
            {"ref": DEM_REF},
            parquet=PARQUET,
            filter_out=None,
            n_bootstrap=0,
        )
        unfiltered.run(pc_align=False)
        assert "no land-cover filter" in unfiltered.summary_plot()._suptitle.get_text()

    def test_bootstrap_off(self, tmp_path):
        bench = DEMBenchmark(
            str(tmp_path),
            {"ref": DEM_REF, "stereo": DEM_STEREO},
            parquet=PARQUET,
            n_bootstrap=0,
        )
        df = bench.run(pc_align=False)
        assert bench.bootstrap is None
        assert df[["nmad_ci_low_m", "p_best"]].isna().all().all()
        assert np.isfinite(df["dh_shared_nmad_m"]).all()
        fig = bench.summary_plot()
        matplotlib.pyplot.close(fig)

    def test_negative_bootstrap_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="n_bootstrap"):
            DEMBenchmark(str(tmp_path), [DEM_STEREO], parquet=PARQUET, n_bootstrap=-1)

    def test_with_pc_align(self, bench, fake_pc_align):
        df = bench.run(pc_align=True, minimum_points=10)
        assert len(fake_pc_align) == 2
        by = df.set_index("label")
        # Ranked and bootstrapped on the aligned residuals, on shared points.
        assert bench.bootstrap["stage"] == "aligned"
        assert np.isfinite(by["dh_aligned_shared_nmad_m"]).all()
        assert np.isfinite(by["nmad_ci_low_m"]).all()
        shared_aligned = bench.dh_aligned["stereo"].reindex(bench.shared_ids)
        assert shared_aligned.notna().all()
        assert by.loc["stereo", "dh_aligned_shared_nmad_m"] == pytest.approx(
            1.4826 * np.median(np.abs(shared_aligned - np.median(shared_aligned)))
        )
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
        translated_fn = os.path.join(
            bench.label_directory("stereo"),
            os.path.basename(DEM_STEREO).replace(".tif", "_pc_align_translated.tif"),
        )
        first = os.path.getmtime(translated_fn)
        # Second run reuses the existing pc_align products: no new subprocess,
        # and the translated DEM, newer than its inputs, is not rewritten.
        df2 = bench.run(pc_align=True, minimum_points=10)
        assert len(fake_pc_align) == 2
        assert os.path.getmtime(translated_fn) == first
        pd.testing.assert_frame_equal(df2, df)
        # A translated DEM older than its pc_align output is rewritten, and
        # no partial copy is left behind.
        os.utime(translated_fn, (first - 10, first - 10))
        bench.run(pc_align=True, minimum_points=10)
        assert os.path.getmtime(translated_fn) > first - 10
        assert not [
            f for f in os.listdir(bench.label_directory("stereo")) if ".partial" in f
        ]

    def test_unreadable_translated_dem_does_not_empty_shared_set(
        self, bench, fake_pc_align, monkeypatch
    ):
        # If the translated copy samples NaN everywhere (unreadable, or being
        # rewritten by a concurrent run), that DEM is scored without
        # alignment and the other DEMs keep their shared points.
        from asp_plot.alignment import Alignment

        real = Alignment.apply_dem_translation

        def far_away(self, output_prefix, output_fn=None):
            real(self, output_prefix=output_prefix, output_fn=output_fn)
            return DEM_FAR if "stereo" in output_fn else output_fn

        monkeypatch.setattr(Alignment, "apply_dem_translation", far_away)
        df = bench.run(pc_align=True, minimum_points=10).set_index("label")
        assert "stereo" not in bench.dh_aligned
        assert "ref" in bench.dh_aligned
        assert np.isnan(df.loc["stereo", "dh_aligned_nmad_m"])
        assert np.isfinite(df.loc["ref", "dh_aligned_shared_nmad_m"])
        assert df["n_shared"].gt(0).all()
        assert len(bench.shared_ids) == df.loc["stereo", "n_shared"]
        # The DEM that lost its alignment is outside the bootstrap: it has no
        # interval, and the figure must not call it "≈ best".
        assert bench.bootstrap["labels"] == ["ref"]
        assert np.isnan(df.loc["stereo", "nmad_vs_best_ci_low_m"])
        fig = bench.summary_plot()
        texts = [t.get_text() for t in fig.axes[2].texts]
        assert sum("best" in t for t in texts) == 1
        assert not any("≈" in t for t in texts)

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
        # A DEM with no points does not empty the shared set; it just has
        # no shared statistics of its own.
        assert df.loc["stereo", "n_shared"] == df.loc["stereo", "n_points"]
        assert np.isnan(df.loc["far", "n_shared"])
        assert np.isnan(df.loc["far", "p_best"])
        assert df.loc["stereo", "p_best"] == 1.0
        assert bench.bootstrap["labels"] == ["stereo"]

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


class TestPlanetaryBlocks:
    """The resampling block of LOLA/MOLA points, without a planetary DEM:
    the tagging runs on the point table alone."""

    @staticmethod
    def _alt(columns):
        from types import SimpleNamespace

        n = 6
        table = pd.DataFrame(
            {
                "lon": np.linspace(0, 1, n),
                "lat": np.linspace(0, 1, n),
                "height": np.zeros(n),
                **columns,
            }
        )
        return SimpleNamespace(planetary_points=table)

    def test_mola_orbit_is_the_block(self, bench):
        # ODE GDS pads its header names with spaces.
        alt = self._alt({"          ORBIT": [11, 11, 12, 12, 12, 13]})
        bench._tag_planetary_points(alt)
        assert list(alt.planetary_points["benchmark_point_id"]) == list(range(6))
        assert bench.block_name == "orbit"
        assert bench.blocks.nunique() == 3
        assert list(bench.blocks.index) == list(range(6))
        assert bench.blocks.iloc[0] == bench.blocks.iloc[1]
        assert bench.blocks.iloc[1] != bench.blocks.iloc[2]

    def test_lola_has_no_block_and_warns_once(self, bench, caplog):
        with caplog.at_level("WARNING", logger="asp_plot.dem_benchmark"):
            for _ in range(3):  # one table per DEM
                alt = self._alt({"Pt_Radius": np.full(6, 1737.4)})
                bench._tag_planetary_points(alt)
                assert "benchmark_point_id" in alt.planetary_points.columns
        assert bench.blocks is None
        assert bench.block_name is None
        warnings = [r for r in caplog.records if "no orbit id" in r.getMessage()]
        assert len(warnings) == 1


class TestOutput:
    def test_save_stats_and_figures(self, bench, fake_pc_align, tmp_path):
        bench.run(pc_align=True, minimum_points=10)
        csv_fn = bench.save_stats(str(tmp_path / "out" / "stats.csv"))
        assert list(pd.read_csv(csv_fn).columns) == STATS_COLUMNS

        fig = bench.summary_plot(save_dir=str(tmp_path), fig_fn="summary.png")
        assert os.path.exists(tmp_path / "summary.png")
        # Best-first ordering by post-alignment NMAD on the shared points.
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        expected = list(bench.stats_df.sort_values("dh_aligned_shared_nmad_m")["label"])
        assert labels == expected
        # Coverage, median, NMAD. IntersectionErr stays in the table only.
        assert len(fig.axes) == 3
        assert "IntersectionErr" not in [ax.get_title() for ax in fig.axes]
        assert np.isfinite(
            bench.stats_df.set_index("label").loc["stereo", "ie_median_m"]
        )
        # The NMAD panel prints the interval and names the best DEM; the
        # subtitle says what the error bars are.
        nmad_texts = [t.get_text() for t in fig.axes[2].texts]
        assert any("best" in t for t in nmad_texts)
        assert all("[" in t for t in nmad_texts)
        assert "resampling" in fig._suptitle.get_text()
        assert "valid in every DEM" in fig._suptitle.get_text()
        matplotlib.pyplot.close(fig)

        fig = bench.histogram_plot(save_dir=str(tmp_path), fig_fn="hist.png")
        assert os.path.exists(tmp_path / "hist.png")
        assert "after pc_align" in fig.axes[0].get_xlabel()
        assert "valid in every DEM" in fig.axes[0].get_xlabel()
        # The legend counts the shared points, not each DEM's own.
        n_shared = int(bench.stats_df["n_shared"].iloc[0])
        for text in fig.axes[0].get_legend().get_texts():
            assert f"n={n_shared}" in text.get_text()
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
                "--n-bootstrap",
                "50",
                "--output-filename",
                "bench.png",
            ],
        )
        assert result.exit_code == 0, result.output
        for fn in ("bench.png", "bench_histogram.png", "bench.csv"):
            assert os.path.exists(tmp_path / fn), fn
        df = pd.read_csv(tmp_path / "bench.csv")
        assert list(df["label"]) == ["reference", "date_time_left_right_1m"]
        assert list(df.columns) == STATS_COLUMNS
        assert np.isfinite(df["nmad_ci_low_m"]).all()
        assert "reference" in result.output
        # The printed table shows the shared-point headline columns.
        assert "dh_shared_nmad_m" in result.output
        assert "p_best" in result.output
        assert "Paired bootstrap (50 replicates" in result.output

    def test_cli_filter_out(self, tmp_path):
        from click.testing import CliRunner

        from asp_plot.cli.dem_benchmark import main

        def run(name, *options):
            result = CliRunner().invoke(
                main,
                [
                    DEM_REF,
                    "--parquet",
                    PARQUET,
                    "--directory",
                    str(tmp_path),
                    "--no-pc-align",
                    "--n-bootstrap",
                    "0",
                    "--output-filename",
                    f"{name}.png",
                    *options,
                ],
            )
            assert result.exit_code == 0, result.output
            return int(pd.read_csv(tmp_path / f"{name}.csv")["n_points"].iloc[0])

        # The comparison below needs both classes in the fixture.
        classes = pd.read_parquet(PARQUET, columns=["esa_worldcover.value"])[
            "esa_worldcover.value"
        ]
        assert classes.isin([80]).any(), "fixture has no water points"
        assert classes.isin(
            [20, 30, 40, 90, 95, 100]
        ).any(), "fixture has no low-vegetation points"

        default = run("default")
        two = run("two", "--filter-out", "water", "--filter-out", "low_vegetation")
        everything = run("everything", "--filter-out", "none")
        assert two < default < everything

        result = CliRunner().invoke(
            main, [DEM_REF, "--parquet", PARQUET, "--filter-out", "tree"]
        )
        assert result.exit_code != 0
        assert "tree" in result.output

        # Giving the option replaces the default: trees alone keeps water.
        trees_only = run("trees_only", "--filter-out", "trees")
        assert trees_only == everything  # the fixture has no tree points
        assert not classes.isin([10]).any()

        # 'none' means no filter, so it cannot sit beside a group.
        result = CliRunner().invoke(
            main,
            [
                DEM_REF,
                "--parquet",
                PARQUET,
                "--filter-out",
                "none",
                "--filter-out",
                "trees",
            ],
        )
        assert result.exit_code != 0
        assert "cannot be combined" in result.output


def _write_dem(path, z, transform, crs, nodata=-9999.0):
    import rasterio as rio

    z = np.asarray(z, dtype="float32")
    with rio.open(
        path,
        "w",
        driver="GTiff",
        width=z.shape[1],
        height=z.shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as ds:
        ds.write(z, 1)
    return str(path)


def _points(xs, ys, crs):
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)


@pytest.fixture
def tilted_plane(tmp_path):
    """A 20° plane rising to the north-east at 10 m pixels, in UTM 4N, with
    one nodata pixel."""
    from rasterio.transform import from_origin

    theta = np.radians(20.0)
    n = 60
    centres = (np.arange(n) + 0.5) * 10.0
    east, north = np.meshgrid(centres, centres[::-1])
    # Gradient of tan(20°), split equally between east and north.
    z = np.tan(theta) / np.sqrt(2) * (east + north)
    z[30, 30] = -9999.0
    fn = _write_dem(
        tmp_path / "plane.tif", z, from_origin(500000, 7900600, 10, 10), "EPSG:32604"
    )
    return fn


@pytest.fixture
def slope_ramp_dem(tmp_path):
    """A DEM over the ICESat-2 fixture whose slope climbs from 0° to 60° and
    drops back every 2 km west to east, so points anywhere on it fall in
    every class."""
    import rasterio as rio
    from rasterio.transform import from_origin

    # Over the stereo DEM, which holds every shared point.
    with rio.open(DEM_STEREO) as ds:
        left, bottom, right, top = ds.bounds
        crs = ds.crs
    res = 20.0
    width = int(np.ceil((right - left) / res)) + 2
    height = int(np.ceil((top - bottom) / res)) + 2
    x = (np.arange(width) + 0.5) * res
    slope = np.radians(60.0) * ((x / 2000.0) % 1.0)
    z = np.tile(np.cumsum(np.tan(slope)) * res, (height, 1))
    return _write_dem(
        tmp_path / "ramp.tif", z, from_origin(left - res, top + res, res, res), crs
    )


class TestSlopeAtPoints:
    def test_plane_matches_its_slope(self, tilted_plane):
        xs = 500000 + np.array([105.0, 255.0, 455.0, 305.0, 305.0, 5.0, 800.0])
        ys = 7900600 - np.array([105.0, 255.0, 455.0, 305.0, 295.0, 5.0, 100.0])
        slope = slope_at_points(tilted_plane, _points(xs, ys, "EPSG:32604"))
        np.testing.assert_allclose(slope[:3], 20.0, atol=1e-4)
        # Next to the nodata pixel, on the DEM edge, and outside it: NaN.
        assert np.isnan(slope[3:]).all()

    def test_points_in_another_crs(self, tilted_plane):
        from pyproj import Transformer

        lon, lat = Transformer.from_crs(
            "EPSG:32604", "EPSG:4326", always_xy=True
        ).transform(500105.0, 7900495.0)
        slope = slope_at_points(tilted_plane, _points([lon], [lat], "EPSG:4326"))
        assert slope[0] == pytest.approx(20.0, abs=1e-4)

    def test_per_point_reads_match_one_box_read(self, tilted_plane, monkeypatch):
        rng = np.random.default_rng(0)
        xs = 500000 + rng.uniform(0, 600, 200)
        ys = 7900600 - rng.uniform(0, 600, 200)
        points = _points(xs, ys, "EPSG:32604")
        boxed = slope_at_points(tilted_plane, points)
        monkeypatch.setattr("asp_plot.terrain.MAX_WINDOW_PIXELS", 1)
        np.testing.assert_array_equal(slope_at_points(tilted_plane, points), boxed)
        assert np.isfinite(boxed).sum() > 100

    def test_geographic_dem(self, tmp_path):
        # Heights rising north at tan(10°), with pixel sizes in degrees at
        # 70° N: the slope needs the pixel converted to meters.
        from rasterio.transform import from_origin

        res = 0.0005
        n = 40
        lat0 = 70.0
        lats = lat0 + (np.arange(n)[::-1] + 0.5) * res
        meters_north = (lats - lat0) * 111694.0  # meridian arc per degree at 70° N
        z = np.tile((np.tan(np.radians(10.0)) * meters_north)[:, None], (1, n))
        fn = _write_dem(
            tmp_path / "geo.tif",
            z,
            from_origin(-156.0, lat0 + n * res, res, res),
            "EPSG:4326",
        )
        slope = slope_at_points(
            fn, _points([-156.0 + 20.5 * res], [lat0 + 20.5 * res], "EPSG:4326")
        )
        assert slope[0] == pytest.approx(10.0, abs=0.02)

    def test_scale_averages_a_fine_dem_first(self, tmp_path):
        # A 20° plane with 5 m of noise on its 10 m pixels: the native slope
        # is that of the noise, the slope of 5 × 5 block means close to the
        # plane's.
        from rasterio.transform import from_origin

        n = 120
        rng = np.random.default_rng(1)
        z = np.tan(np.radians(20.0)) * np.arange(n) * 10.0 + rng.normal(0, 5, (n, n))
        fn = _write_dem(
            tmp_path / "rough.tif",
            z,
            from_origin(500000, 7900600, 10, 10),
            "EPSG:32604",
        )
        xs = 500000 + rng.uniform(200, 1000, 80)
        ys = 7900600 - rng.uniform(200, 1000, 80)
        points = _points(xs, ys, "EPSG:32604")
        native = slope_at_points(fn, points)
        averaged = slope_at_points(fn, points, scale=50)
        assert np.abs(native - 20).mean() > 3 * np.abs(averaged - 20).mean()
        np.testing.assert_allclose(averaged, 20.0, atol=2.0)
        # A point's slope does not depend on the other points passed
        np.testing.assert_array_equal(
            slope_at_points(fn, points.iloc[40:], scale=50), averaged[40:]
        )
        # A scale under two pixels leaves the DEM as it is
        np.testing.assert_array_equal(slope_at_points(fn, points, scale=19), native)
        # Points off the DEM
        far = _points([400000.0], [7900000.0], "EPSG:32604")
        assert np.isnan(slope_at_points(fn, far, scale=50)).all()

    def test_pixel_size_and_label(self, tilted_plane):
        assert slope_pixel_size(tilted_plane) == (10.0, False)
        assert slope_pixel_size(tilted_plane, scale=30) == (30.0, True)
        # Whole pixels only: 25 m holds two 10 m pixels
        assert slope_pixel_size(tilted_plane, scale=25) == (20.0, True)
        assert slope_pixel_size(tilted_plane, scale=10) == (10.0, False)
        assert slope_source_label(tilted_plane, 10) == "Slope from plane.tif (10 m)"
        assert (
            slope_source_label(tilted_plane, 30, name="MVS")
            == "Slope from MVS (averaged to 30 m)"
        )

    def test_no_points(self, tilted_plane):
        assert slope_at_points(tilted_plane, _points([], [], "EPSG:32604")).size == 0


class TestSlopeBreakdown:
    @staticmethod
    def _bench(tmp_path, slope_dem, **kwargs):
        kwargs.setdefault("n_bootstrap", 50)
        return DEMBenchmark(
            str(tmp_path / "work"),
            {"ref": DEM_REF, "stereo": DEM_STEREO},
            parquet=PARQUET,
            slope_dem=slope_dem,
            **kwargs,
        )

    def test_validation(self, tmp_path, slope_ramp_dem):
        with pytest.raises(ValueError, match="neither one of the DEM labels"):
            self._bench(tmp_path, "nope.tif")
        for bins in [(5,), (0, 10, 5), (-1, 5), (0, 95)]:
            with pytest.raises(ValueError, match="slope_bins"):
                self._bench(tmp_path, slope_ramp_dem, slope_bins=bins)
        by_label = self._bench(tmp_path, "ref")
        assert by_label.slope_dem == DEM_REF
        assert by_label.slope_dem_name == "ref"
        by_path = self._bench(tmp_path, slope_ramp_dem)
        assert by_path.slope_dem_name == "ramp.tif"
        # A fine slope DEM is averaged to 30 m unless slope_scale says otherwise
        assert by_label.slope_scale == 30
        assert self._bench(tmp_path, "ref", slope_scale=0).slope_scale is None

    def test_off_without_a_logged_reference_dem(self, tmp_path, capsys):
        bench = self._bench(tmp_path, "auto")
        out = capsys.readouterr().out
        assert "No slope breakdown" in out
        assert "no stereo logs beside the DEMs name a reference DEM" in out
        assert bench.slope_dem is None
        bench.run(pc_align=False)
        assert bench.stats_by_slope_df is None
        assert bench.slope is None
        with pytest.raises(ValueError, match="slope_dem"):
            bench.slope_plot()

    def test_reference_dem_from_logs(self, tmp_path):
        assert reference_dem_from_logs(DEM_STEREO) == LOGGED_REFERENCE_DEM
        # No stereo logs in the folder
        assert reference_dem_from_logs(DEM_REF) is None
        # Stereo logs that name no DEM (images not mapprojected)
        assert reference_dem_from_logs(DEM_FAR) is None

        # A relative path in the log is relative to the folder above the DEM's
        stereo = tmp_path / "proc" / "stereo"
        stereo.mkdir(parents=True)
        (tmp_path / "proc" / "ref").mkdir()
        shutil.copy(DEM_REF, tmp_path / "proc" / "ref" / "cop30.tif")
        shutil.copy(DEM_REF, stereo / "run-DEM.tif")
        with open("tests/test_data/stereo/log-stereo_pprc.txt") as f:
            log = f.read()
        assert LOGGED_REFERENCE_DEM in log
        (stereo / "run-log-stereo_pprc.txt").write_text(
            log.replace(LOGGED_REFERENCE_DEM, "ref/cop30.tif")
        )
        assert reference_dem_from_logs(str(stereo / "run-DEM.tif")) == str(
            tmp_path / "proc" / "ref" / "cop30.tif"
        )

    def test_auto_uses_the_logged_reference_dem(
        self, tmp_path, slope_ramp_dem, logged_reference_dems, capsys
    ):
        # DEM_REF has no logs and so no say
        logged_reference_dems[DEM_STEREO] = slope_ramp_dem
        bench = self._bench(tmp_path, "auto")
        assert bench.slope_dem == os.path.realpath(slope_ramp_dem)
        assert bench.slope_dem_name == "ramp.tif"
        assert "Slope DEM (the reference DEM in the stereo logs)" in (
            capsys.readouterr().out
        )
        bench.run(pc_align=False)
        assert bench.stats_by_slope_df is not None

        assert self._bench(tmp_path, None).slope_dem is None

    def test_auto_skips_when_logs_disagree_or_file_is_missing(
        self, tmp_path, slope_ramp_dem, logged_reference_dems, capsys
    ):
        logged_reference_dems[DEM_STEREO] = slope_ramp_dem
        logged_reference_dems[DEM_REF] = DEM_FAR
        assert self._bench(tmp_path, "auto").slope_dem is None
        assert "different reference DEMs" in capsys.readouterr().out

        logged_reference_dems.clear()
        logged_reference_dems[DEM_STEREO] = str(tmp_path / "elsewhere" / "cop30.tif")
        assert self._bench(tmp_path, "auto").slope_dem is None
        assert "does not exist" in capsys.readouterr().out

    def test_auto_falls_back_to_the_reference_label(
        self, tmp_path, logged_reference_dems, capsys
    ):
        bench = self._bench(tmp_path, "auto", reference="ref")
        assert bench.slope_dem == DEM_REF
        assert bench.slope_dem_name == "ref"
        assert "Slope DEM: reference='ref', since no stereo logs" in (
            capsys.readouterr().out
        )
        # A logged reference DEM still comes first
        logged_reference_dems[DEM_STEREO] = DEM_FAR
        assert self._bench(tmp_path, "auto", reference="ref").slope_dem == (
            os.path.realpath(DEM_FAR)
        )

    def test_slope_dem_off_the_points_gives_no_breakdown(self, tmp_path, capsys):
        bench = self._bench(tmp_path, DEM_FAR)
        bench.run(pc_align=False)
        assert bench.stats_by_slope_df is None
        assert bench.slope is None
        assert "gives no slope at any of the shared points" in capsys.readouterr().out

    def test_classes_and_stats(self, tmp_path, slope_ramp_dem):
        bench = self._bench(tmp_path, slope_ramp_dem)
        bench.run(pc_align=False)
        df = bench.stats_by_slope_df
        assert list(df.columns) == SLOPE_STATS_COLUMNS
        assert len(df) == 2 * 5
        assert list(df["slope_bin"].unique()) == [
            "0–5°",
            "5–10°",
            "10–20°",
            "20–30°",
            "30–90°",
        ]
        # One slope per shared point, the same classes for both DEMs.
        assert bench.slope.index.equals(bench.shared_ids)
        counts = df.groupby("slope_bin", sort=False)["n_points"].agg(["min", "max"])
        assert (counts["min"] == counts["max"]).all()
        assert (counts["min"] > 0).all(), "the ramp should put points in every class"
        assert counts["min"].sum() == np.isfinite(bench.slope).sum()
        # Each class is scored on exactly its points.
        for row in df.itertuples():
            in_class = (bench.slope >= row.slope_min_deg) & (
                bench.slope < row.slope_max_deg
            )
            dh = bench.dh[row.label].reindex(bench.shared_ids)[in_class.to_numpy()]
            assert row.dh_shared_median_m == pytest.approx(np.median(dh))
            assert row.n_blocks == bench.blocks.reindex(dh.index).nunique()
            assert row.nmad_ci_low_m <= row.dh_shared_nmad_m <= row.nmad_ci_high_m
            assert row.median_ci_low_m <= row.dh_shared_median_m <= row.median_ci_high_m
        assert df["dh_aligned_shared_nmad_m"].isna().all()

    def test_thin_class_is_counted_not_scored(self, tmp_path, slope_ramp_dem):
        # Find a slope value held by fewer than SLOPE_MIN_POINTS points and
        # put a class around just that value.
        probe = self._bench(tmp_path, slope_ramp_dem, n_bootstrap=0)
        probe.run(pc_align=False)
        values, counts = np.unique(probe.slope.dropna().to_numpy(), return_counts=True)
        i = next(
            i
            for i in range(1, len(values) - 1)
            if counts[i] < SLOPE_MIN_POINTS and 0 < values[i - 1]
        )
        lo = (values[i - 1] + values[i]) / 2
        hi = (values[i] + values[i + 1]) / 2
        bench = self._bench(tmp_path, slope_ramp_dem, slope_bins=(0, lo, hi, 90))
        bench.run(pc_align=False)
        df = bench.stats_by_slope_df
        n = df.drop_duplicates("slope_bin")["n_points"].to_numpy()
        assert 0 < n[1] < SLOPE_MIN_POINTS
        thin = df[df["slope_min_deg"] == lo]
        scored = [c for c in SLOPE_STATS_COLUMNS if c.endswith("_m")]
        assert thin[scored].isna().all().all()
        assert df[df["slope_min_deg"] == 0][scored[:2]].notna().all().all()
        fig = bench.slope_plot()
        ticks = [t.get_text() for t in fig.axes[-1].get_xticklabels()]
        assert "(too few)" in ticks[1]
        assert "(too few)" not in ticks[0]
        matplotlib.pyplot.close(fig)

    def test_points_without_slope_are_left_out(self, tmp_path):
        # The ref fixture is 5 x 5 pixels: points in its edge pixels get no
        # slope and fall in no class.
        bench = self._bench(tmp_path, "ref", slope_bins=(0, 1, 90), n_bootstrap=0)
        bench.run(pc_align=False)
        df = bench.stats_by_slope_df
        assert df["nmad_ci_low_m"].isna().all()
        n_with = np.isfinite(bench.slope).sum()
        assert 0 < n_with
        assert df.drop_duplicates("slope_bin")["n_points"].sum() == n_with

    def test_with_pc_align_and_outputs(self, tmp_path, slope_ramp_dem, fake_pc_align):
        bench = self._bench(tmp_path, slope_ramp_dem, title="slope test")
        bench.run(pc_align=True, minimum_points=10)
        df = bench.stats_by_slope_df
        assert np.isfinite(df["dh_aligned_shared_nmad_m"]).all()
        # Intervals are on the aligned stage, the one the bootstrap ranks.
        assert (df["nmad_ci_low_m"] <= df["dh_aligned_shared_nmad_m"]).all()
        assert (df["dh_aligned_shared_nmad_m"] <= df["nmad_ci_high_m"]).all()

        csv_fn = bench.save_stats(str(tmp_path / "out" / "bench.csv"))
        by_slope = pd.read_csv(tmp_path / "out" / "bench_by_slope.csv")
        assert bench.slope_csv_path(csv_fn).endswith("bench_by_slope.csv")
        assert list(by_slope.columns) == SLOPE_STATS_COLUMNS

        fig = bench.slope_plot(save_dir=str(tmp_path), fig_fn="slope.png")
        assert os.path.exists(tmp_path / "slope.png")
        median_ax, ax = fig.axes
        assert "median" in median_ax.get_ylabel()
        assert "NMAD" in ax.get_ylabel()
        ticks = [t.get_text() for t in ax.get_xticklabels()]
        assert len(ticks) == 5
        assert all("n = " in t and "tracks" in t for t in ticks)
        legend = [t.get_text() for t in median_ax.get_legend().get_texts()]
        assert legend == list(bench._sorted_stats(True)["label"])
        title = fig._suptitle.get_text()
        assert "slope test" in title
        assert "horizontal pc_align shift 0.61 m" in title  # one fixture log for both
        assert "within each class" in title
        assert "after pc_align" in ax.get_ylabel()
        assert ax.get_xlabel() == "Slope from ramp.tif (20 m)"
        matplotlib.pyplot.close(fig)

        # Before alignment: no error bars (the bootstrap ran on aligned).
        fig = bench.slope_plot(aligned=False)
        assert "before pc_align" in fig.axes[0].get_ylabel()
        assert "within each class" not in fig._suptitle.get_text()
        matplotlib.pyplot.close(fig)

    def test_cli(self, tmp_path, slope_ramp_dem):
        from click.testing import CliRunner

        from asp_plot.cli.dem_benchmark import main

        result = CliRunner().invoke(
            main,
            [
                DEM_REF,
                DEM_STEREO,
                "--parquet",
                PARQUET,
                "--directory",
                str(tmp_path),
                "--no-pc-align",
                "--n-bootstrap",
                "20",
                "--slope-dem",
                slope_ramp_dem,
                "--slope-bins",
                "0,10,30,90",
                "--output-filename",
                "bench.png",
            ],
        )
        assert result.exit_code == 0, result.output
        for fn in ("bench_slope.png", "bench_by_slope.csv"):
            assert os.path.exists(tmp_path / fn), fn
        df = pd.read_csv(tmp_path / "bench_by_slope.csv")
        assert list(df["slope_bin"].unique()) == ["0–10°", "10–30°", "30–90°"]
        assert "dh_shared_median_m by slope class" in result.output
        assert "dh_shared_nmad_m by slope class" in result.output
        assert "n points" in result.output

        result = CliRunner().invoke(
            main, [DEM_REF, "--parquet", PARQUET, "--slope-bins", "0,a"]
        )
        assert result.exit_code != 0
        assert "--slope-bins" in result.output

    def test_cli_auto_and_none(self, tmp_path, slope_ramp_dem, logged_reference_dems):
        from click.testing import CliRunner

        from asp_plot.cli.dem_benchmark import main

        logged_reference_dems[DEM_STEREO] = slope_ramp_dem
        args = [
            DEM_REF,
            DEM_STEREO,
            "--parquet",
            PARQUET,
            "--no-pc-align",
            "--n-bootstrap",
            "20",
        ]
        result = CliRunner().invoke(main, args + ["--directory", str(tmp_path / "a")])
        assert result.exit_code == 0, result.output
        assert "by slope class (slope from ramp.tif)" in result.output
        assert os.path.exists(tmp_path / "a" / "dem_benchmark_slope.png")

        result = CliRunner().invoke(
            main, args + ["--directory", str(tmp_path / "b"), "--slope-dem", "none"]
        )
        assert result.exit_code == 0, result.output
        assert "by slope class" not in result.output
        assert not os.path.exists(tmp_path / "b" / "dem_benchmark_slope.png")
