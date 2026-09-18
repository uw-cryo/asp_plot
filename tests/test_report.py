import os

import matplotlib
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from asp_plot.report import (
    FIGURE_MAX_DPI,
    MM_PER_INCH,
    AlignmentReportPage,
    ReportMetadata,
    ReportSection,
    _downsampled_for_page,
    _fmt_sig,
    compile_report,
)

matplotlib.use("Agg")


@pytest.fixture
def dummy_image(tmp_path):
    """Write a tiny PNG that the PDF renderer can embed."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot([1, 2, 3], [1, 4, 2])
    path = str(tmp_path / "dummy.png")
    fig.savefig(path, dpi=60)
    plt.close(fig)
    return path


def _minimal_params_dict():
    return {
        "processing_timestamp": "2026-04-17",
        "asp_version": "3.7.0-alpha",
        "bundle_adjust": "bundle_adjust ...",
        "bundle_adjust_run_time": "0:05",
        "stereo": "parallel_stereo ...",
        "stereo_run_time": "0:28",
        "point2dem": "point2dem ...",
        "point2dem_run_time": "0:01",
        "reference_dem": "/path/to/ref.tif",
    }


def test_fmt_sig_formatting():
    assert _fmt_sig(1.2345) == "1.23"
    assert _fmt_sig(12.5) == "12.5"
    assert _fmt_sig(123.456) == "123"
    assert _fmt_sig(0) == "0"
    assert _fmt_sig(-3.1415) == "-3.14"
    assert _fmt_sig(float("nan")) == "n/a"


def test_metadata_table_wraps_long_values():
    """Long values (e.g. the joined acquisition dates of a tri-stereo collect)
    must wrap inside the DEM-summary table instead of overflowing the page."""
    from asp_plot.report import ASPReportPDF, ReportMetadata, _add_metadata_table

    dates = [
        "2021-11-07 10:29:11 UTC",
        "2021-11-07 10:29:28 UTC",
        "2021-11-07 10:29:44 UTC",
    ]

    def table_height(acquisition_dates):
        pdf = ASPReportPDF(report_title="t")
        pdf.add_page()
        y0 = pdf.get_y()
        _add_metadata_table(
            pdf,
            ReportMetadata(
                dem_filename="run-DEM.tif", acquisition_dates=acquisition_dates
            ),
        )
        return pdf.get_y() - y0

    # Three joined dates exceed the value-column width, so their row must be
    # taller than the single-date row (i.e. the value wrapped onto new lines).
    assert table_height(dates) > table_height(dates[:1])


def test_compile_report_processing_parameters_on_page_two(dummy_image, tmp_path):
    """Processing Parameters should land on page 2, not the last page."""
    out = str(tmp_path / "layout.pdf")
    compile_report(
        sections=[
            ReportSection(title="Dummy", image_path=dummy_image, caption="."),
        ],
        processing_parameters_dict=_minimal_params_dict(),
        report_pdf_path=out,
        report_title="Layout Test",
        report_metadata=ReportMetadata(
            dem_filename="test-DEM.tif",
            dem_dimensions=(100, 100),
            dem_gsd_m=1.0,
            dem_crs="EPSG:32611",
        ),
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_compile_report_with_alignment_pages(dummy_image, tmp_path):
    """AlignmentReportPage (success, no_improvement, insufficient) should render."""
    sections = [
        ReportSection(title="Dummy", image_path=dummy_image),
        AlignmentReportPage(
            title="DEM Alignment with ICESat-2",
            parameters={"processing_level": "all", "improvement_threshold_pct": 5.0},
            stats_row={
                "median_beg": 1.2,
                "nmad_beg": 0.9,
                "rmse_beg": 3.1,
                "median_end": 0.3,
                "nmad_end": 0.25,
                "rmse_end": 1.4,
                "north_shift": -0.1,
                "east_shift": 0.3,
                "down_shift": -1.2,
                "translation_magnitude": 1.3,
            },
            status_message="Success. Aligned DEM written.",
            image_path=dummy_image,
            caption="Pre/post landcover distributions.",
        ),
        AlignmentReportPage(
            title="DEM Alignment with ICESat-2",
            parameters={"processing_level": "all"},
            stats_row={"p50_beg": 1.2, "p50_end": 1.18},
            status_message="No significant improvement. Aligned DEM removed.",
        ),
        AlignmentReportPage(
            title="DEM Alignment with ICESat-2",
            status_message="Alignment skipped: insufficient points.",
        ),
    ]
    out = str(tmp_path / "alignment.pdf")
    compile_report(
        sections=sections,
        processing_parameters_dict=_minimal_params_dict(),
        report_pdf_path=out,
        report_title="Alignment Test",
        report_metadata=ReportMetadata(dem_filename="test-DEM.tif"),
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


class TestAlignmentStatsTables:
    def test_split_pairs_beg_end_and_keeps_translation_in_order(self):
        from asp_plot.report import _split_alignment_stats

        row = {
            "median_beg": 5.71,
            "nmad_beg": 3.21,
            "median_end": 2.01,
            "nmad_end": 2.07,
            "north_shift": -0.09,
            "east_shift": -1.18,
            "down_shift": 4.76,
            "translation_magnitude": 4.90,
        }
        stats, translation = _split_alignment_stats(row)
        assert stats == [("Median", 5.71, 2.01), ("NMAD", 3.21, 2.07)]
        assert translation == [
            ("North", -0.09),
            ("East", -1.18),
            ("Down", 4.76),
            ("Magnitude |T|", 4.90),
        ]

    def test_split_pre_370_percentiles_and_unknown_keys(self):
        from asp_plot.report import _split_alignment_stats

        row = {"p50_beg": 1.0, "p50_end": 0.5, "|T|": 0.3, "p84_beg": 2.0}
        stats, translation = _split_alignment_stats(row)
        assert stats == [("p50", 1.0, 0.5)]
        # an unpaired _beg and an unknown key fall through, labelled as-is
        assert translation == [("|T|", 0.3), ("p84_beg", 2.0)]

    @pytest.mark.parametrize(
        "before, after, expected",
        [
            (5.71, 2.01, "-64.8%"),
            (153.0, 153.3, "+0.2%"),
            (0.0, 1.0, "n/a"),
            (float("nan"), 1.0, "n/a"),
            ("x", 1.0, "n/a"),
        ],
    )
    def test_fmt_pct_change(self, before, after, expected):
        from asp_plot.report import _fmt_pct_change

        assert _fmt_pct_change(before, after) == expected

    def test_lone_table_renders(self, tmp_path):
        """Only-translation or only-stats rows each render as a single
        full-width table without error."""
        for stats_row in (
            {"north_shift": 0.1, "east_shift": 0.2, "down_shift": 0.3},
            {"median_beg": 1.0, "median_end": 0.5},
        ):
            out = str(tmp_path / f"lone_{len(stats_row)}.pdf")
            compile_report(
                sections=[
                    AlignmentReportPage(
                        title="Lone table", stats_row=stats_row, status_message="."
                    )
                ],
                processing_parameters_dict=_minimal_params_dict(),
                report_pdf_path=out,
            )
            assert os.path.exists(out) and os.path.getsize(out) > 0


@pytest.fixture
def oversized_image(tmp_path):
    """A figure whose pixel width far exceeds what a Letter page can show.

    figsize=(16, 4) at 220 dpi is 3520 px wide; placed across the 185.9 mm
    usable width of a Letter page that is ~480 effective dpi, which is the
    shape of the real multi-panel report figures.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    ax.plot([1, 2, 3], [1, 4, 2])
    path = str(tmp_path / "wide.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def test_downsample_caps_resolution_at_placed_width(oversized_image):
    """An over-resolved figure comes back resized to exactly the dpi ceiling."""
    render_w_mm = 185.9  # Letter width minus 15 mm margins
    resized = _downsampled_for_page(oversized_image, render_w_mm, 200)

    assert not isinstance(resized, str), "oversized figure should be resized"
    expected_px = int(round(render_w_mm / MM_PER_INCH * 200))
    assert resized.width == expected_px

    with Image.open(oversized_image) as original:
        assert resized.width < original.width
        # Aspect ratio preserved to within a rounded pixel.
        assert (
            abs(resized.height / resized.width - original.height / original.width)
            < 0.01
        )


def test_downsample_leaves_small_figures_alone(dummy_image):
    """A figure already under the ceiling is embedded as-is, not re-encoded."""
    assert _downsampled_for_page(dummy_image, 185.9, 200) == dummy_image


@pytest.mark.parametrize("max_dpi", [0, None, -1])
def test_downsample_disabled(oversized_image, max_dpi):
    """Falsy or non-positive ceilings pass the original path straight through."""
    assert _downsampled_for_page(oversized_image, 185.9, max_dpi) == oversized_image


def test_compile_report_downsampling_shrinks_pdf(oversized_image, tmp_path):
    """The dpi cap is wired into compile_report and measurably shrinks output."""
    sections = [
        ReportSection(title=f"Wide {i}", image_path=oversized_image, caption=".")
        for i in range(3)
    ]
    capped = str(tmp_path / "capped.pdf")
    uncapped = str(tmp_path / "uncapped.pdf")
    for out, dpi in ((capped, FIGURE_MAX_DPI), (uncapped, 0)):
        compile_report(
            sections=sections,
            processing_parameters_dict=_minimal_params_dict(),
            report_pdf_path=out,
            report_title="DPI Test",
            figure_max_dpi=dpi,
        )

    assert os.path.getsize(capped) < os.path.getsize(uncapped)
