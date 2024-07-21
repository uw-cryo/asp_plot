import glob
import logging
import os

import geoutils as gu
import matplotlib.colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from markdown_pdf import MarkdownPdf, Section
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal
from rasterio.windows import Window

logger = logging.getLogger(__name__)


def glob_file(directory, *patterns, all_files=False):
    for pattern in patterns:
        files = glob.glob(os.path.join(directory, pattern))
        if files:
            if all_files:
                return files
            else:
                return files[0]
    logger.warning(
        f"Could not find {patterns} in {directory}. Some plots may be missing."
    )
    return None


def show_existing_figure(filename):
    if os.path.exists(filename):
        img = mpimg.imread(filename)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
    else:
        print(f"Figure not found: {filename}")


def save_figure(fig, save_dir=None, fig_fn=None, dpi=150):
    if save_dir or not fig_fn:
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, fig_fn)
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {file_path}")
    else:
        raise ValueError("\n\nPlease provide a save directory and figure filename.\n\n")


def compile_report(
    plots_directory, processing_parameters_dict, report_pdf_path, report_title=None
):
    from PIL import Image

    files = [f for f in os.listdir(plots_directory) if f.endswith(".png")]
    files.sort()

    # Convert .png files to .jpg with 95% quality
    compressed_files = []
    for file in files:
        png_path = os.path.join(plots_directory, file)
        jpg_file = file.replace(".png", ".jpg")
        jpg_path = os.path.join(plots_directory, jpg_file)

        with Image.open(png_path) as img:
            img = img.convert("RGB")
            img.save(jpg_path, "JPEG", quality=95)

        compressed_files.append(jpg_file)

    processing_date = processing_parameters_dict["processing_timestamp"]

    if report_title is None:
        report_title = os.path.basename(os.path.dirname(report_pdf_path))

    report_title = (
        f"# ASP Report\n\n## {report_title:}\n\nProcessed on: {processing_date:}"
    )

    ba_string = f"### Bundle Adjust:\n\n{processing_parameters_dict['bundle_adjust']:}"
    stereo_string = f"### Stereo:\n\n{processing_parameters_dict['stereo']:}"
    point2dem_string = f"### point2dem:\n\n{processing_parameters_dict['point2dem']:}"

    pdf = MarkdownPdf()

    pdf.add_section(Section(f"{report_title:}\n\n"))
    pdf.add_section(
        Section(
            f"## Processing Parameters\n\n{ba_string:}\n\n{stereo_string}\n\n{point2dem_string}\n\n"
        )
    )
    plots = "".join([f"![]({file})\n\n" for file in compressed_files])
    pdf.add_section(Section(f"## Plots\n\n{plots:}", root=plots_directory))

    pdf.save(report_pdf_path)

    # cleanup temporary JPEG files
    for file in compressed_files:
        jpg_path = os.path.join(plots_directory, file)
        os.remove(jpg_path)


def get_xml_tag(xml, tag, all=False):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml)
    if all:
        elem = tree.findall(".//%s" % tag)
        elem = [i.text for i in elem]
    else:
        elem = tree.find(".//%s" % tag)
        elem = elem.text

    return elem


class ColorBar:
    def __init__(self, perc_range=(2, 98), symm=False):
        self.perc_range = perc_range
        self.symm = symm

    def get_clim(self, input):
        try:
            clim = np.percentile(input.compressed(), self.perc_range)
        except:
            clim = np.percentile(input, self.perc_range)
        if self.symm:
            clim = self.symm_clim(clim)
        return clim

    def find_common_clim(self, inputs):
        clims = []
        for input in inputs:
            clim = self.get_clim(input)
            clims.append(clim)

        clim_min = np.min([clim[0] for clim in clims])
        clim_max = np.max([clim[1] for clim in clims])
        clim = (clim_min, clim_max)
        if self.symm:
            clim = self.symm_clim(clim)
        return clim

    def symm_clim(self, clim):
        abs_max = np.max(np.abs(clim))
        return (-abs_max, abs_max)

    def get_cbar_extend(self, input, clim=None):
        if clim is None:
            clim = self.get_clim(input)
        extend = "both"
        if input.min() >= clim[0] and input.max() <= clim[1]:
            extend = "neither"
        elif input.min() >= clim[0] and input.max() > clim[1]:
            extend = "max"
        elif input.min() < clim[0] and input.max() <= clim[1]:
            extend = "min"
        return extend


class Raster:
    def __init__(self, fn):
        self.fn = fn
        self.ds = rio.open(fn)

    def read_array(self, b=1, extent=False):
        a = self.ds.read(b, masked=True)
        ndv = self.get_ndv()
        ma = np.ma.fix_invalid(np.ma.masked_equal(a, ndv))
        out = ma
        if extent:
            extent = rio.plot.plotting_extent(self.ds)
            out = (ma, extent)
        return out

    def get_ndv(self):
        ndv = self.ds.nodatavals[0]
        if ndv is None:
            ndv = self.ds.read(1, window=Window(0, 0, 1, 1)).squeeze()
        return ndv

    def get_gsd(self):
        gsd = self.ds.transform[0]
        return gsd

    def hillshade(self):
        hs_fn = os.path.splitext(self.fn)[0] + "_hs.tif"
        if os.path.exists(hs_fn):
            hillshade = Raster(hs_fn).read_array()
        else:
            gdal_ds = gdal.Open(self.fn)
            hs_ds = gdal.DEMProcessing(
                "", gdal_ds, "hillshade", format="MEM", computeEdges=True
            )
            hillshade = np.ma.masked_equal(hs_ds.ReadAsArray(), 0)
        return hillshade

    def compute_difference(self, second_fn):
        fn_list = [self.fn, second_fn]
        outdir = os.path.dirname(os.path.abspath(self.fn))

        outprefix = (
            os.path.splitext(os.path.split(self.fn)[1])[0]
            + "_"
            + os.path.splitext(os.path.split(second_fn)[1])[0]
        )

        rasters = gu.raster.load_multiple_rasters(fn_list, ref_grid=1)
        diff = rasters[1] - rasters[0]
        dst_fn = os.path.join(outdir, outprefix + "_diff.tif")
        diff.save(dst_fn)
        return diff.data


class Plotter:
    def __init__(
        self,
        clim_perc=(2, 98),
        lognorm=False,
        title=None,
    ):
        self.clim_perc = clim_perc
        self.lognorm = lognorm
        self.title = title
        self.cb = ColorBar(perc_range=self.clim_perc)

    def plot_array(
        self,
        ax,
        array,
        clim=None,
        cmap="inferno",
        add_cbar=True,
        cbar_label=None,
        alpha=1,
    ):
        if clim is None:
            clim = self.cb.get_clim(array)

        im = ax.imshow(
            array,
            cmap=cmap,
            clim=clim,
            alpha=alpha,
            interpolation="none",
        )

        if add_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad="2%")
            plt.colorbar(
                im,
                cax=cax,
                ax=ax,
                extend=self.cb.get_cbar_extend(array, clim),
            )
            cax.set_ylabel(cbar_label)

        ax.set_facecolor("0.5")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self.title)

    def plot_geodataframe(
        self, ax, gdf, column_name, clim=None, cmap="inferno", cbar_label=None
    ):
        if clim is None:
            clim = self.cb.get_clim(gdf[column_name])
        vmin, vmax = clim

        if self.lognorm:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        gdf.plot(
            ax=ax,
            column=column_name,
            cmap=cmap,
            norm=norm,
            s=1,
            legend=True,
            legend_kwds={"label": cbar_label},
        )
