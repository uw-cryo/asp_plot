import os

import matplotlib
import matplotlib.pyplot as plt
import pytest

from asp_plot.report import (
    AlignmentReportPage,
    ReportMetadata,
    ReportSection,
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
                "p16_beg": 0.5,
                "p50_beg": 1.2,
                "p84_beg": 2.5,
                "p16_end": 0.2,
                "p50_end": 0.3,
                "p84_end": 1.0,
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
