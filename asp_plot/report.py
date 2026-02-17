import logging
import os
import textwrap
from dataclasses import dataclass

from fpdf import FPDF

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
    """

    dem_dimensions: tuple = (0, 0)
    dem_gsd_m: float = 0.0
    dem_crs: str = ""
    dem_nodata_percent: float = 0.0
    dem_elevation_range: tuple = (0, 0)
    dem_filename: str = ""
    reference_dem: str = ""


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
    pdf.ln(10)

    if report_metadata is not None:
        _add_metadata_table(pdf, report_metadata)

    # ---- Figure sections ----
    for i, section in enumerate(sections, start=1):
        section.figure_number = i
        if not os.path.exists(section.image_path):
            logger.warning(f"Image not found, skipping: {section.image_path}")
            continue

        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, section.title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        usable_width = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(section.image_path, x=pdf.l_margin, w=usable_width)

        if section.caption:
            pdf.ln(3)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, f"Figure {section.figure_number}: {section.caption}")

    # ---- Processing Parameters page ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Processing Parameters", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    _add_runtime_table(pdf, processing_parameters_dict)
    pdf.ln(6)

    ref_dem = processing_parameters_dict.get("reference_dem", "")
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
        cmd = processing_parameters_dict.get(key, "")
        if cmd:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, f"{label} Command:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Courier", "", 7)
            wrapped = textwrap.fill(cmd, width=120)
            pdf.multi_cell(0, 4, wrapped)
            pdf.ln(4)

    pdf.output(report_pdf_path)


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
