import logging
import math
import os
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from fpdf import FPDF
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    """A section of the PDF report containing a figure with title and caption.

    Attributes
    ----------
    title : str
        Section heading displayed above the figure.
    image_path : str
        Absolute path to the PNG image file.
    caption : str
        Caption text displayed below the figure.
    figure_number : int
        Auto-assigned by compile_report().
    """

    title: str
    image_path: str
    caption: str = ""
    figure_number: int = 0


@dataclass
class AlignmentReportPage:
    """A PDF page for the pc_align-vs-ICESat-2 alignment workflow.

    Each page carries (optionally) a small kwargs table, a single-row
    alignment-stats table, a description paragraph, a status/message block,
    and an optional figure with caption below. Rendered by compile_report
    alongside ReportSection entries. Body text blocks are rendered
    left-aligned (not justified) to avoid large inter-word gaps.

    Attributes
    ----------
    title : str
        Page heading.
    parameters : dict
        Alignment kwargs passed to ``Altimetry.align_and_evaluate``. Rendered
        as a small two-column table above the stats table. Use an empty dict
        to skip.
    stats_row : dict
        Single-row alignment statistics (e.g. p16_beg/p50_beg/... from
        ``pc_align_report``). Rendered as a horizontal 1-row table with
        column headers. Values are formatted to two significant figures.
        Use an empty dict to skip.
    description : str
        Long-form explanation of pc_align and the meaning of each column in
        the parameters and stats tables. Rendered between the stats table
        and the status message. Empty string to skip.
    status_message : str
        Short status paragraph (e.g. path to aligned DEM, or a note that
        alignment was skipped / produced no significant improvement).
    image_path : str or None
        Optional absolute path to a PNG figure rendered below the tables.
    caption : str
        Optional caption shown below the figure when ``image_path`` is set.
    figure_number : int
        Auto-assigned by compile_report().
    """

    title: str
    parameters: dict = field(default_factory=dict)
    stats_row: dict = field(default_factory=dict)
    description: str = ""
    status_message: str = ""
    image_path: Optional[str] = None
    caption: str = ""
    figure_number: int = 0


@dataclass
class ReportMetadata:
    """Metadata about the output DEM for the report title page.

    Attributes
    ----------
    dem_dimensions : tuple
        (width, height) in pixels.
    dem_gsd_m : float
        Ground sample distance in meters.
    dem_crs : str
        Coordinate reference system string (e.g. "EPSG:32616").
    dem_nodata_percent : float
        Percentage of nodata pixels.
    dem_elevation_range : tuple
        (min, max) elevation in meters.
    dem_filename : str
        DEM filename.
    reference_dem : str
        Reference DEM path or description.
    acquisition_dates : list of str
        Scene acquisition date strings (e.g. "2017-07-31 19:07:28 UTC") when
        recoverable from scene metadata. Empty list if not found.
    """

    dem_dimensions: tuple = (0, 0)
    dem_gsd_m: float = 0.0
    dem_crs: str = ""
    dem_nodata_percent: float = 0.0
    dem_elevation_range: tuple = (0, 0)
    dem_filename: str = ""
    reference_dem: str = ""
    acquisition_dates: list = field(default_factory=list)


class ASPReportPDF(FPDF):
    """FPDF subclass with custom header and footer for ASP reports."""

    def __init__(self, report_title="ASP Output Quality Report"):
        super().__init__(orientation="P", unit="mm", format="Letter")
        self.report_title = report_title
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(15, 15, 15)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 6, self.report_title, align="L")
            self.ln(2)
            self.set_draw_color(200, 200, 200)
            self.line(15, self.get_y(), self.w - 15, self.get_y())
            self.ln(4)
            self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def compile_report(
    sections,
    processing_parameters_dict,
    report_pdf_path,
    report_title="ASP Output Quality Report",
    report_metadata=None,
    report_command=None,
):
    """
    Compile a PDF report with ASP processing results and plots.

    Creates a structured PDF report with a title page, figure sections
    with captions, and a processing parameters appendix.

    Parameters
    ----------
    sections : list of ReportSection
        Ordered list of report sections, each containing a title,
        image path, and optional caption.
    processing_parameters_dict : dict
        Dictionary containing processing parameters from ASP logs.
    report_pdf_path : str
        Output path for the PDF report.
    report_title : str, optional
        Title for the report. Default is "ASP Output Quality Report".
    report_metadata : ReportMetadata, optional
        DEM metadata for the title page summary table. Default is None.
    report_command : str, optional
        The asp_plot CLI command used to generate this report. Default is None.

    Returns
    -------
    None
        Generates a PDF report at the specified path.

    Notes
    -----
    Required keys in processing_parameters_dict:
    - processing_timestamp: When the processing was performed
    - reference_dem: Path to reference DEM used
    - bundle_adjust: Bundle adjustment command
    - bundle_adjust_run_time: Time to run bundle adjustment
    - stereo: Stereo command
    - stereo_run_time: Time to run stereo
    - point2dem: Point2dem command
    - point2dem_run_time: Time to run point2dem
    """
    pdf = ASPReportPDF(report_title=report_title)
    pdf.alias_nb_pages()

    # ---- Title page ----
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font("Helvetica", "B", 22)
    pdf.multi_cell(0, 12, report_title, align="C")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 11)
    processing_date = processing_parameters_dict.get("processing_timestamp", "")
    pdf.cell(
        0,
        8,
        f"Processed on: {processing_date}",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    asp_version = processing_parameters_dict.get("asp_version", "")
    if asp_version:
        pdf.cell(
            0,
            8,
            f"ASP version: {asp_version}",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )

    from asp_plot import __version__

    pdf.cell(
        0,
        8,
        f"asp_plot version: {__version__}",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(10)

    if report_metadata is not None:
        _add_metadata_table(pdf, report_metadata)

    # ---- Processing Parameters page (page 2) ----
    _add_processing_parameters_page(pdf, processing_parameters_dict, report_command)

    # ---- Figure sections ----
    for i, section in enumerate(sections, start=1):
        section.figure_number = i
        if isinstance(section, AlignmentReportPage):
            _render_alignment_report_page(pdf, section)
            continue

        if not os.path.exists(section.image_path):
            logger.warning(f"Image not found, skipping: {section.image_path}")
            continue

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, section.title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        _render_figure_with_caption(
            pdf, section.image_path, section.caption, section.figure_number
        )

    pdf.output(report_pdf_path)


def _render_figure_with_caption(pdf, image_path, caption, figure_number):
    """Render a figure scaled to the remaining page, with optional caption.

    Parameters
    ----------
    pdf : ASPReportPDF
    image_path : str
    caption : str
    figure_number : int
    """
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    if caption:
        caption_text = f"Figure {figure_number}: {caption}"
        estimated_lines = max(1, -(-len(caption_text) // 80))
        caption_reserve = estimated_lines * 5 + 8
    else:
        caption_reserve = 0
    usable_height = pdf.h - pdf.get_y() - pdf.b_margin - caption_reserve

    with Image.open(image_path) as img:
        img_w, img_h = img.size
    aspect = img_h / img_w
    render_w = usable_width
    render_h = render_w * aspect
    if render_h > usable_height:
        render_h = usable_height
        render_w = render_h / aspect

    pdf.image(
        image_path,
        x=pdf.l_margin + (usable_width - render_w) / 2,
        w=render_w,
    )

    if caption:
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, f"Figure {figure_number}: {caption}")


def _render_alignment_report_page(pdf, page):
    """Render an AlignmentReportPage: header + optional tables + status + figure.

    Parameters
    ----------
    pdf : ASPReportPDF
    page : AlignmentReportPage
    """
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, page.title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if page.parameters:
        _add_alignment_parameters_table(pdf, page.parameters)
        pdf.ln(3)

    if page.stats_row:
        _add_alignment_stats_row_table(pdf, page.stats_row)
        pdf.ln(3)

    if page.description:
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 4.5, page.description, align="L")
        pdf.ln(2)

    if page.status_message:
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 5, page.status_message, align="L")
        pdf.ln(3)

    if page.image_path and os.path.exists(page.image_path):
        _render_figure_with_caption(
            pdf, page.image_path, page.caption, page.figure_number
        )


def _fmt_sig(x):
    """Format a numeric value compactly with roughly two significant figures.

    Uses fixed-point notation with 2 decimals for |x| < 10, 1 decimal for
    10 <= |x| < 100, and 0 decimals above. Returns "n/a" for non-finite
    values.
    """
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "n/a"
    if xf == 0:
        return "0"
    ax = abs(xf)
    if ax < 10:
        return f"{xf:.2f}"
    if ax < 100:
        return f"{xf:.1f}"
    return f"{xf:.0f}"


def _add_alignment_parameters_table(pdf, parameters):
    """Render the alignment kwargs table (two columns: parameter, value).

    Parameters
    ----------
    pdf : ASPReportPDF
    parameters : dict
    """
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Alignment Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(col_w, 6, "Parameter", border=1, fill=True)
    pdf.cell(col_w, 6, "Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for key, val in parameters.items():
        pdf.cell(col_w, 6, str(key), border=1)
        pdf.cell(col_w, 6, str(val), border=1, new_x="LMARGIN", new_y="NEXT")


_ALIGNMENT_STATS_DISPLAY_LABELS = {
    "north_shift": "N_shift",
    "east_shift": "E_shift",
    "down_shift": "D_shift",
    "translation_magnitude": "|T|",
}


def _add_alignment_stats_row_table(pdf, stats_row):
    """Render a single-row horizontal alignment stats table.

    Each key becomes a column header; the corresponding value becomes the
    single data row (formatted to two significant figures). Long pc_align
    field names are shortened via ``_ALIGNMENT_STATS_DISPLAY_LABELS`` so the
    headers fit inside the table columns.

    Parameters
    ----------
    pdf : ASPReportPDF
    stats_row : dict
        Ordered dict-like of ``{column_name: value}``.
    """
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Alignment Statistics (m)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    keys = list(stats_row.keys())
    if not keys:
        return

    usable_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = usable_w / len(keys)

    pdf.set_font("Helvetica", "B", 7)
    pdf.set_fill_color(220, 220, 220)
    for k in keys:
        label = _ALIGNMENT_STATS_DISPLAY_LABELS.get(k, str(k))
        pdf.cell(col_w, 6, label, border=1, fill=True, align="C")
    pdf.ln(6)

    pdf.set_font("Helvetica", "", 8)
    for k in keys:
        pdf.cell(col_w, 6, _fmt_sig(stats_row[k]), border=1, align="C")
    pdf.ln(6)


def _add_processing_parameters_page(pdf, params, report_command):
    """Add the Processing Parameters page (runtime table + commands).

    Parameters
    ----------
    pdf : ASPReportPDF
        The PDF document.
    params : dict
        Processing parameters dictionary from ProcessingParameters.from_log_files().
    report_command : str or None
        The asp_plot CLI command used to generate this report.
    """
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Processing Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    _add_runtime_table(pdf, params)
    pdf.ln(6)

    ref_dem = params.get("reference_dem", "")
    if ref_dem:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Reference DEM:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 7)
        pdf.multi_cell(0, 4, ref_dem)
        pdf.ln(4)

    for key, label in [
        ("bundle_adjust", "Bundle Adjust"),
        ("stereo", "Stereo"),
        ("point2dem", "point2dem"),
    ]:
        cmd = params.get(key, "")
        if cmd:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"{label} Command:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", "", 7)
            wrapped = textwrap.fill(cmd, width=120)
            pdf.multi_cell(0, 4, wrapped)
            pdf.ln(4)

    if report_command:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Report Generation Command:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", "", 7)
        wrapped = textwrap.fill(report_command, width=120)
        pdf.multi_cell(0, 4, wrapped)
        pdf.ln(4)


def _add_metadata_table(pdf, metadata):
    """Add DEM metadata summary table to the PDF.

    Parameters
    ----------
    pdf : ASPReportPDF
        The PDF document.
    metadata : ReportMetadata
        DEM metadata to display.
    """
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "DEM Summary", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    rows = []
    if metadata.dem_filename:
        rows.append(("DEM File", metadata.dem_filename))
    if metadata.dem_dimensions != (0, 0):
        w, h = metadata.dem_dimensions
        rows.append(("Dimensions (px)", f"{w} x {h}"))
    if metadata.dem_gsd_m:
        rows.append(("GSD (m)", f"{metadata.dem_gsd_m:.2f}"))
    if metadata.dem_crs:
        rows.append(("CRS", metadata.dem_crs))
    if metadata.dem_nodata_percent:
        rows.append(("Nodata (%)", f"{metadata.dem_nodata_percent:.1f}"))
    if metadata.dem_elevation_range != (0, 0):
        lo, hi = metadata.dem_elevation_range
        rows.append(("Elevation Range (m)", f"{lo:.1f} to {hi:.1f}"))
    if metadata.reference_dem:
        rows.append(("Reference DEM", metadata.reference_dem))
    if metadata.acquisition_dates:
        if len(metadata.acquisition_dates) == 1:
            rows.append(("Acquisition Date", metadata.acquisition_dates[0]))
        else:
            rows.append(("Acquisition Dates", "; ".join(metadata.acquisition_dates)))

    if not rows:
        return

    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 2
    table_x = pdf.l_margin

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.set_x(table_x)
    pdf.cell(col_w, 7, "Property", border=1, fill=True)
    pdf.cell(col_w, 7, "Value", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for prop, val in rows:
        pdf.set_x(table_x)
        pdf.cell(col_w, 7, prop, border=1)
        pdf.cell(col_w, 7, str(val), border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)


def _add_runtime_table(pdf, params):
    """Add runtime summary table to the PDF.

    Parameters
    ----------
    pdf : ASPReportPDF
        The PDF document.
    params : dict
        Processing parameters dictionary.
    """
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Runtime Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    col_w = (pdf.w - pdf.l_margin - pdf.r_margin) / 2

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(220, 220, 220)
    pdf.cell(col_w, 7, "Step", border=1, fill=True)
    pdf.cell(col_w, 7, "Runtime", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    for key, label in [
        ("bundle_adjust_run_time", "Bundle Adjust"),
        ("stereo_run_time", "Stereo"),
        ("point2dem_run_time", "point2dem"),
    ]:
        runtime = params.get(key, "N/A")
        pdf.cell(col_w, 7, label, border=1)
        pdf.cell(col_w, 7, str(runtime), border=1, new_x="LMARGIN", new_y="NEXT")
