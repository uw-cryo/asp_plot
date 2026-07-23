"""Tests for the declarative report pipeline (issue #128).

These characterize the orchestration that used to live inline in the
~880-line ``cli/asp_plot.py::main()``: the section registry, its gating
predicates, figure numbering, and the exact sequence of report sections
produced for the Earth and planetary paths. All plotting/IO is monkeypatched,
so the tests run without ASP, SlideRule, or network access.
"""

import dataclasses
import os

import numpy as np
import pandas as pd
import pytest

import asp_plot.report_pipeline as rp
from asp_plot.report import AlignmentReportPage
from asp_plot.report_pipeline import (
    REPORT_SECTIONS,
    ReportConfig,
    ReportContext,
    _resolve_plot_altimetry,
    run_report,
)

DEM = "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"


# ---------------------------------------------------------------------------
# ReportConfig <-> CLI option parity
# ---------------------------------------------------------------------------


class TestReportConfig:
    def test_defaults_match_cli_options(self):
        """Every Click option must map to a ReportConfig field with the same
        default, so the CLI's keyword splat stays correct."""
        from asp_plot.cli.asp_plot import main

        cfg_fields = {f.name: f for f in dataclasses.fields(ReportConfig)}
        # report_command is synthesized by the CLI, not a Click option.
        click_params = {p.name: p for p in main.params}

        for name, param in click_params.items():
            assert name in cfg_fields, f"{name} missing from ReportConfig"
            assert (
                cfg_fields[name].default == param.default
            ), f"default mismatch for {name}"

        # The only config field with no matching Click option:
        extra = set(cfg_fields) - set(click_params)
        assert extra == {"report_command"}


# ---------------------------------------------------------------------------
# Registry shape + predicates
# ---------------------------------------------------------------------------


def _ctx(**overrides):
    """Minimal ReportContext for predicate tests (no plotting)."""
    config = overrides.pop("config", ReportConfig())
    defaults = dict(
        config=config,
        plots_directory="/tmp/plots",
        report_pdf_path="/tmp/r.pdf",
        report_title="r",
        map_crs="EPSG:4326",
        ctx_kwargs={},
        stereo_plotter=None,
        asp_dem=DEM,
        plot_altimetry=config.plot_altimetry,
    )
    defaults.update(overrides)
    return ReportContext(**defaults)


class TestRegistry:
    def test_section_order(self):
        assert [s.name for s in REPORT_SECTIONS] == [
            "input_scenes",
            "stereo_geometry",
            "match_points",
            "bundle_adjust",
            "disparity",
            "dem_results",
            "detailed_hillshade",
            "altimetry",
        ]

    def test_always_on_sections(self):
        ctx = _ctx()
        for name in (
            "input_scenes",
            "match_points",
            "disparity",
            "dem_results",
            "detailed_hillshade",
        ):
            spec = next(s for s in REPORT_SECTIONS if s.name == name)
            assert spec.enabled(ctx) is True

    def test_stereo_geometry_predicate(self):
        spec = next(s for s in REPORT_SECTIONS if s.name == "stereo_geometry")
        assert spec.enabled(_ctx(config=ReportConfig(plot_geometry=True))) is True
        assert spec.enabled(_ctx(config=ReportConfig(plot_geometry=False))) is False

    def test_bundle_adjust_predicate(self):
        spec = next(s for s in REPORT_SECTIONS if s.name == "bundle_adjust")
        cfg = ReportConfig(bundle_adjust_directory="ba")
        assert spec.enabled(_ctx(config=cfg)) is True
        assert spec.enabled(_ctx(config=ReportConfig())) is False

    def test_altimetry_predicate_reads_resolved_flag(self):
        spec = next(s for s in REPORT_SECTIONS if s.name == "altimetry")
        assert spec.enabled(_ctx(plot_altimetry=True)) is True
        assert spec.enabled(_ctx(plot_altimetry=False)) is False


# ---------------------------------------------------------------------------
# Stereo geometry section fan-out (N-scene runs save several figures)
# ---------------------------------------------------------------------------


class TestStereoGeometryFanOut:
    """stereo_geom_plot returns the saved figure names: two scenes save
    exactly fig_fn, N scenes save an overview plus one figure per pair, and
    each saved figure becomes its own report section."""

    def _build(self, monkeypatch, saved_factory):
        class _Geom:
            def __init__(self, *a, **k):
                pass

            def stereo_geom_plot(self, save_dir=None, fig_fn=None):
                return saved_factory(fig_fn)

        monkeypatch.setattr(rp, "StereoGeometryPlotter", _Geom)
        return rp._build_stereo_geometry(_ctx(plots_directory="/tmp/plots"))

    def test_multi_figure_run_fans_out_into_sections(self, monkeypatch):
        sections = self._build(
            monkeypatch,
            lambda fig_fn: [
                f"{fig_fn[:-4]}_overview.png",
                f"{fig_fn[:-4]}_pairA.png",
                f"{fig_fn[:-4]}_pairB.png",
            ],
        )
        assert [s.title for s in sections] == [
            "Stereo Geometry",
            "Stereo Geometry (continued)",
            "Stereo Geometry (continued)",
        ]
        # The caption belongs to the first figure only.
        assert sections[0].caption
        assert sections[1].caption == "" and sections[2].caption == ""
        assert [os.path.basename(s.image_path) for s in sections] == [
            "00_overview.png",
            "00_pairA.png",
            "00_pairB.png",
        ]

    def test_plotter_returning_none_falls_back_to_fig_fn(self, monkeypatch):
        sections = self._build(monkeypatch, lambda fig_fn: None)
        assert [s.title for s in sections] == ["Stereo Geometry"]
        assert os.path.basename(sections[0].image_path) == "00.png"


class TestStereoGeometryScoping:
    """The geometry section is scoped to the cameras named in the stereo
    command when they can be recovered from the run's log, falling back to
    directory-based discovery otherwise."""

    def _build(self, monkeypatch, camera_files, fail_on_inputs=False):
        constructed = []

        class _Geom:
            def __init__(self, directory=None, add_basemap=True, inputs=None):
                if fail_on_inputs and inputs is not None:
                    raise ValueError("no matching sensor")
                constructed.append({"directory": directory, "inputs": inputs})

            def stereo_geom_plot(self, save_dir=None, fig_fn=None):
                return [fig_fn]

        monkeypatch.setattr(rp, "StereoGeometryPlotter", _Geom)
        monkeypatch.setattr(
            rp, "camera_files_from_stereo_run", lambda d, s: camera_files
        )
        rp._build_stereo_geometry(_ctx(plots_directory="/tmp/plots"))
        return constructed

    def test_scopes_to_recovered_cameras(self, monkeypatch):
        constructed = self._build(monkeypatch, ["left.xml", "right.xml"])
        assert constructed == [
            {"directory": ReportConfig().directory, "inputs": ["left.xml", "right.xml"]}
        ]

    def test_falls_back_when_unrecoverable(self, monkeypatch):
        constructed = self._build(monkeypatch, None)
        assert constructed == [{"directory": ReportConfig().directory, "inputs": None}]

    def test_falls_back_when_sensor_detection_fails(self, monkeypatch):
        constructed = self._build(
            monkeypatch, ["left.xml", "right.xml"], fail_on_inputs=True
        )
        assert constructed == [{"directory": ReportConfig().directory, "inputs": None}]


# ---------------------------------------------------------------------------
# --plot_icesat deprecation alias
# ---------------------------------------------------------------------------


class TestResolvePlotAltimetry:
    def test_none_passes_through_plot_altimetry(self):
        assert _resolve_plot_altimetry(ReportConfig(plot_altimetry=True)) is True
        assert _resolve_plot_altimetry(ReportConfig(plot_altimetry=False)) is False

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("False", False),
            ("false", False),
            ("0", False),
            ("no", False),
            ("True", True),
            ("anything", True),
        ],
    )
    def test_deprecated_string_overrides(self, value, expected):
        with pytest.warns(DeprecationWarning):
            out = _resolve_plot_altimetry(
                ReportConfig(plot_altimetry=True, plot_icesat=value)
            )
        assert out is expected

    def test_deprecated_bool_overrides(self):
        with pytest.warns(DeprecationWarning):
            assert (
                _resolve_plot_altimetry(
                    ReportConfig(plot_altimetry=True, plot_icesat=False)
                )
                is False
            )


# ---------------------------------------------------------------------------
# End-to-end orchestration with everything monkeypatched
# ---------------------------------------------------------------------------


class _FakeRaster:
    def __init__(self, fn):
        self.fn = fn

        class _DS:
            width = 100
            height = 200
            crs = "EPSG:32610"

        self.ds = _DS()

    def get_epsg_code(self):
        return 32610

    def get_gsd(self):
        return 1.0

    def read_array(self):
        return np.ma.array(np.ones((4, 4), dtype=float), mask=np.zeros((4, 4), bool))


class _Recorder:
    """Records plotting calls (which fig_fn was drawn, in order)."""

    def __init__(self):
        self.fig_fns = []
        self.calls = []

    def record(self, name, kwargs):
        self.calls.append(name)
        if "fig_fn" in kwargs:
            self.fig_fns.append(kwargs["fig_fn"])


class _FakeStereoPlotter:
    def __init__(self, rec):
        self._rec = rec
        self.dem_fn = DEM
        self.title = None
        self.detailed_hillshade_clips = None

    def _mk(name):
        def method(self, **kwargs):
            self._rec.record(name, kwargs)

        return method

    plot_match_points = _mk("plot_match_points")
    plot_disparity = _mk("plot_disparity")
    plot_dem_results = _mk("plot_dem_results")
    plot_detailed_hillshade = _mk("plot_detailed_hillshade")


def _make_align_result(status):
    df = pd.DataFrame([{"key": "all", "p50_beg": 1.0, "p50_end": 0.5, "|T|": 0.3}])
    return type(
        "AR",
        (),
        {
            "status": status,
            "parameters_used": {"processing_level": "all"},
            "message": f"msg-{status}",
            "alignment_report_df": df,
        },
    )()


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Patch every plotter/IO dependency in report_pipeline; capture the
    sections handed to compile_report and the order of figure filenames."""
    rec = _Recorder()
    captured = {}

    monkeypatch.setattr(rp, "Raster", _FakeRaster)
    monkeypatch.setattr(rp, "get_acquisition_dates", lambda *a, **k: ["2020-01-01"])
    monkeypatch.setattr(rp, "StereoPlotter", lambda *a, **k: _FakeStereoPlotter(rec))

    class _FakeScene:
        def __init__(self, *a, **k):
            pass

        def plot_scenes(self, **kwargs):
            rec.record("plot_scenes", kwargs)

    class _FakeGeom:
        def __init__(self, *a, **k):
            pass

        def stereo_geom_plot(self, **kwargs):
            rec.record("stereo_geom_plot", kwargs)
            # Mirror the real two-scene behavior: exactly fig_fn is saved.
            return [kwargs["fig_fn"]]

    monkeypatch.setattr(rp, "ScenePlotter", _FakeScene)
    monkeypatch.setattr(rp, "StereoGeometryPlotter", _FakeGeom)

    class _FakePP:
        def __init__(self, *a, **k):
            pass

        def from_log_files(self):
            return {}

    monkeypatch.setattr(rp, "ProcessingParameters", _FakePP)

    def _fake_compile(sections, params, path, **kwargs):
        captured["sections"] = sections
        captured["compile_kwargs"] = kwargs

    monkeypatch.setattr(rp, "compile_report", _fake_compile)
    monkeypatch.setattr(rp, "write_selections_yaml", lambda *a, **k: None)
    monkeypatch.setattr(rp, "FigureSelections", lambda **k: dict(k))
    # Leave the plots dir in place so cleanup doesn't matter; redirect rmtree.
    monkeypatch.setattr(rp.shutil, "rmtree", lambda *a, **k: None)

    return rec, captured


def _titles(sections):
    return [s.title for s in sections]


class TestRunReportEarth:
    @pytest.fixture
    def fake_icesat(self, monkeypatch, harness):
        rec, captured = harness
        monkeypatch.setattr(rp, "detect_planetary_body", lambda dem: "earth")

        class _FakeAltimetry:
            def __init__(self, **k):
                pass

            def load_atl06sr_from_parquet(self, p):
                return False

            def request_atl06sr_multi_processing(self, **k):
                rec.record("request_atl06sr", k)

            def filter_esa_worldcover(self, **k):
                pass

            def atl06sr_to_dem_dh(self, **k):
                pass

            def get_altimetry_selections(self, key):
                return {"profile_track": {"track": 1}, "segments": [0, 1]}

            def mapview_plot_atl06sr_to_dem(self, **kwargs):
                rec.record("mapview_atl06sr", kwargs)

            def histogram_by_landcover(self, **kwargs):
                rec.record("hist_landcover", kwargs)

            def plot_atl06sr_dem_profile(self, **kwargs):
                rec.record("profile", kwargs)

            def plot_best_worst_segments(self, **kwargs):
                rec.record("segments", kwargs)

            def align_and_evaluate(self, **k):
                return _make_align_result("success")

        monkeypatch.setattr(rp, "Altimetry", _FakeAltimetry)
        return rec, captured

    def test_earth_success_section_sequence(self, fake_icesat):
        rec, captured = fake_icesat
        cfg = ReportConfig(
            directory="tests/test_data",
            stereo_directory="stereo",
            map_crs="EPSG:32610",
            add_basemap=False,
        )
        run_report(cfg)
        assert _titles(captured["sections"]) == [
            "Input Scenes",
            "Stereo Geometry",
            "Match Points",
            "Disparity",
            "DEM Results",
            "Detailed Hillshade",
            "ICESat-2 ATL06-SR Map",
            "ICESat-2 ATL06-SR Histogram",
            "ICESat-2 ATL06-SR Profile",
            "ICESat-2 ATL06-SR Agreement Segments",
            "DEM Alignment with ICESat-2",
            "ICESat-2 ATL06-SR Histogram (Aligned DEM)",
            "ICESat-2 ATL06-SR Profile (Aligned DEM)",
            "ICESat-2 ATL06-SR Agreement Segments (Aligned DEM)",
        ]
        # Figures are numbered sequentially in draw order, no gaps.
        assert rec.fig_fns == [f"{i:02}.png" for i in range(len(rec.fig_fns))]
        # The alignment page is an AlignmentReportPage, not a figure section.
        align = next(
            s for s in captured["sections"] if s.title == "DEM Alignment with ICESat-2"
        )
        assert isinstance(align, AlignmentReportPage)

    def test_plot_geometry_false_drops_section_and_renumbers(self, fake_icesat):
        rec, captured = fake_icesat
        cfg = ReportConfig(
            directory="tests/test_data",
            map_crs="EPSG:32610",
            add_basemap=False,
            plot_geometry=False,
        )
        run_report(cfg)
        assert "Stereo Geometry" not in _titles(captured["sections"])
        # Still contiguous numbering even though a section was skipped.
        assert rec.fig_fns == [f"{i:02}.png" for i in range(len(rec.fig_fns))]
        # Input Scenes (00) then Match Points (01) -- the skipped geometry
        # figure number is not consumed.
        assert rec.fig_fns[0] == "00.png"

    def test_plot_altimetry_false_skips_altimetry(self, fake_icesat):
        rec, captured = fake_icesat
        cfg = ReportConfig(
            directory="tests/test_data",
            map_crs="EPSG:32610",
            add_basemap=False,
            plot_altimetry=False,
        )
        run_report(cfg)
        titles = _titles(captured["sections"])
        assert not any("ICESat-2" in t for t in titles)
        assert titles[-1] == "Detailed Hillshade"

    def test_pc_align_false_stops_after_base_altimetry(self, fake_icesat):
        rec, captured = fake_icesat
        cfg = ReportConfig(
            directory="tests/test_data",
            map_crs="EPSG:32610",
            add_basemap=False,
            pc_align=False,
        )
        run_report(cfg)
        titles = _titles(captured["sections"])
        assert "ICESat-2 ATL06-SR Agreement Segments" in titles
        assert not any("Aligned DEM" in t for t in titles)
        assert "DEM Alignment with ICESat-2" not in titles


class TestRunReportPlanetary:
    @pytest.fixture
    def fake_mars(self, monkeypatch, harness, tmp_path):
        rec, captured = harness
        monkeypatch.setattr(rp, "detect_planetary_body", lambda dem: "mars")

        class _FakeAltimetry:
            def __init__(self, **k):
                pass

            def load_planetary_csv(self, p):
                rec.record("load_planetary_csv", {})

            def planetary_to_dem_dh(self):
                pass

            def mapview_plot_planetary_to_dem(self, **kwargs):
                rec.record("mapview_planetary", kwargs)

            def histogram_planetary_to_dem(self, **kwargs):
                rec.record("hist_planetary", kwargs)

            def align_and_evaluate_planetary(self):
                return _make_align_result("success")

        monkeypatch.setattr(rp, "Altimetry", _FakeAltimetry)
        csv = tmp_path / "mola.csv"
        csv.write_text("x")
        return rec, captured, str(csv)

    def test_mars_success_section_sequence(self, fake_mars):
        rec, captured, csv = fake_mars
        cfg = ReportConfig(
            directory="tests/test_data",
            map_crs="IAU:49900",
            add_basemap=False,
            altimetry_csv=csv,
        )
        run_report(cfg)
        assert _titles(captured["sections"]) == [
            "Input Scenes",
            "Stereo Geometry",
            "Match Points",
            "Disparity",
            "DEM Results",
            "Detailed Hillshade",
            "MOLA Altimetry Map",
            "MOLA Altimetry Histogram",
            "DEM Alignment with MOLA",
            "MOLA Altimetry Map (Aligned DEM)",
            "MOLA Altimetry Histogram (Aligned DEM)",
        ]
        assert rec.fig_fns == [f"{i:02}.png" for i in range(len(rec.fig_fns))]

    def test_mars_without_csv_skips_altimetry_plots(self, fake_mars):
        rec, captured, csv = fake_mars
        cfg = ReportConfig(
            directory="tests/test_data",
            map_crs="IAU:49900",
            add_basemap=False,
            altimetry_csv=None,
        )
        run_report(cfg)
        titles = _titles(captured["sections"])
        assert not any("MOLA" in t for t in titles)
        assert titles[-1] == "Detailed Hillshade"


class TestRunReportReturn:
    def test_returns_pdf_path(self, monkeypatch, harness):
        rec, captured = harness
        monkeypatch.setattr(rp, "detect_planetary_body", lambda dem: "earth")
        cfg = ReportConfig(
            directory="tests/test_data",
            map_crs="EPSG:32610",
            add_basemap=False,
            plot_altimetry=False,
            report_filename="my_report.pdf",
            stereo_directory="stereo",
        )
        out = run_report(cfg)
        assert out.endswith("my_report.pdf")
        assert captured["compile_kwargs"]["report_title"] == "test_data"
