import glob
import logging
import os
import subprocess

import contextily as ctx
import geoutils as gu
import matplotlib.colors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import rioxarray
from markdown_pdf import MarkdownPdf, Section
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osgeo import gdal
from rasterio.windows import Window, from_bounds

logger = logging.getLogger(__name__)


def glob_file(directory, *patterns, all_files=False):
    """
    Find files matching pattern(s) in a directory.

    Searches a directory for files matching one or more glob patterns.
    By default, returns the first matching file found. If all_files is True,
    returns all matching files.

    Parameters
    ----------
    directory : str
        Directory to search in
    *patterns : str
        One or more glob patterns (e.g., "*.tif", "*DEM.tif")
    all_files : bool, optional
        If True, return all matching files; if False, return only the first match.
        Default is False.

    Returns
    -------
    str or list or None
        If all_files is False, returns the first matching file path or None if no matches
        If all_files is True, returns a list of matching file paths or None if no matches

    Examples
    --------
    >>> first_tif = glob_file("/path/to/dir", "*.tif")
    >>> all_tifs = glob_file("/path/to/dir", "*.tif", all_files=True)
    >>> dem_file = glob_file("/path/to/dir", "*-DEM.tif", "*_dem.tif")
    """
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
    """
    Display an existing figure from a file.

    Loads and displays an image file using matplotlib.

    Parameters
    ----------
    filename : str
        Path to the image file

    Returns
    -------
    None
        Displays the figure in the current matplotlib figure

    Notes
    -----
    If the file does not exist, a message is printed and no figure is displayed.
    """
    if os.path.exists(filename):
        img = mpimg.imread(filename)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
    else:
        print(f"Figure not found: {filename}")


def save_figure(fig, save_dir=None, fig_fn=None, dpi=150):
    """
    Save a matplotlib figure to a file.

    Saves a figure to the specified directory with the given filename.
    Creates the directory if it doesn't exist.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    save_dir : str, optional
        Directory to save the figure in
    fig_fn : str, optional
        Filename for the saved figure
    dpi : int, optional
        Resolution in dots per inch, default is 150

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If save_dir or fig_fn is not provided

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> save_figure(fig, save_dir='plots', fig_fn='line_plot.png')
    """
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
    """
    Compile a PDF report with ASP processing results and plots.

    Creates a structured PDF report containing processing parameters and
    generated plots from ASP processing. The plots are converted from PNG
    to JPEG for better compression in the PDF.

    Parameters
    ----------
    plots_directory : str
        Directory containing plot files (PNG format)
    processing_parameters_dict : dict
        Dictionary containing processing parameters from ASP logs
    report_pdf_path : str
        Output path for the PDF report
    report_title : str, optional
        Title for the report. If None, uses the parent directory name

    Returns
    -------
    None
        Generates a PDF report at the specified path

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

    The function converts PNG files to temporary JPG files for the report,
    then deletes the temporary files afterward.
    """
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
    reference_dem_string = (
        f"### Reference DEM:\n\n`{processing_parameters_dict['reference_dem']:}`"
    )
    ba_string = f"### Bundle Adjust ({processing_parameters_dict['bundle_adjust_run_time']:}):\n\n`{processing_parameters_dict['bundle_adjust']:}`"
    stereo_string = f"### Stereo ({processing_parameters_dict['stereo_run_time']:}):\n\n`{processing_parameters_dict['stereo']:}`"
    point2dem_string = f"### point2dem ({processing_parameters_dict['point2dem_run_time']}):\n\n`{processing_parameters_dict['point2dem']:}`"

    pdf = MarkdownPdf()

    pdf.add_section(Section(f"{report_title:}\n\n"))
    pdf.add_section(
        Section(
            f"## Processing Parameters\n\n{reference_dem_string:}\n\n{ba_string:}\n\n{stereo_string}\n\n{point2dem_string}\n\n"
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
    """
    Extract value(s) from XML tag(s).

    Parses an XML file and extracts the content of specified tag(s).

    Parameters
    ----------
    xml : str
        Path to XML file
    tag : str
        XML tag to extract
    all : bool, optional
        If True, find all occurrences of the tag; if False, find first occurrence
        Default is False

    Returns
    -------
    str or list
        If all=False: string content of the first matching tag
        If all=True: list of string contents for all matching tags

    Raises
    ------
    ValueError
        If the tag is not found in the XML file

    Examples
    --------
    >>> satid = get_xml_tag("path/to/file.xml", "SATID")
    >>> ephemeris = get_xml_tag("path/to/file.xml", "EPHEMLIST", all=True)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml)
    if all:
        elem = tree.findall(".//%s" % tag)
        if not elem:
            raise ValueError(f"Tag '{tag}' not found in {xml}")
        elem = [i.text for i in elem]
    else:
        elem = tree.find(".//%s" % tag)
        if elem is None:
            raise ValueError(f"Tag '{tag}' not found in {xml}")
        elem = elem.text

    return elem


def run_subprocess_command(command):
    """
    Run a subprocess command and stream output.

    Executes a shell command using subprocess, streaming its
    output in real time to the console.

    Parameters
    ----------
    command : list or str
        Command to execute as a list of arguments or a single string

    Returns
    -------
    int
        Return code from the command (0 for success)

    Notes
    -----
    This function prints command output in real time and indicates
    whether command execution was successful.

    Examples
    --------
    >>> run_subprocess_command(["ls", "-la"])
    >>> run_subprocess_command("dg_mosaic --skip-tif-gen --output-prefix output_file input_files")
    """
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line.strip())

    process.stdout.close()
    return_code = process.wait()
    if return_code == 0:
        print("\nCommand executed successfully.\n")
    else:
        print("\nCommand failed.\n")

    return return_code


class ColorBar:
    """
    Utility class for managing colorbar limits and normalization.

    This class handles color scaling for data visualization, including
    percentile-based limits, symmetric color mapping, and logarithmic normalization.

    Attributes
    ----------
    perc_range : tuple
        Percentile range (min, max) for color limits, default is (2, 98)
    symm : bool
        Whether to use symmetric color limits, default is False
    clim : tuple or None
        Current color limits (min, max), None until calculated

    Examples
    --------
    >>> cb = ColorBar(perc_range=(5, 95), symm=True)
    >>> clim = cb.get_clim(data)
    >>> norm = cb.get_norm(lognorm=False)
    >>> im = ax.imshow(data, norm=norm)
    >>> plt.colorbar(im, extend=cb.get_cbar_extend(data))
    """

    def __init__(self, perc_range=(2, 98), symm=False):
        """
        Initialize the ColorBar.

        Parameters
        ----------
        perc_range : tuple, optional
            Percentile range (min, max) for color limits, default is (2, 98)
        symm : bool, optional
            Whether to use symmetric color limits, default is False
        """
        self.perc_range = perc_range
        self.symm = symm
        self.clim = None

    def get_clim(self, input):
        """
        Calculate color limits based on data percentiles.

        Parameters
        ----------
        input : array_like
            Input data array (can be masked)

        Returns
        -------
        tuple
            Color limits (min, max)
        """
        try:
            clim = np.nanpercentile(input.compressed(), self.perc_range)
        except:
            clim = np.nanpercentile(input, self.perc_range)
        self.clim = clim
        if self.symm:
            self.clim = self.symm_clim()
        return self.clim

    def find_common_clim(self, inputs):
        """
        Find common color limits across multiple inputs.

        Parameters
        ----------
        inputs : list
            List of input data arrays

        Returns
        -------
        tuple
            Common color limits (min, max)
        """
        clims = []
        for input in inputs:
            clim = self.get_clim(input)
            clims.append(clim)

        clim_min = np.min([clim[0] for clim in clims])
        clim_max = np.max([clim[1] for clim in clims])
        clim = (clim_min, clim_max)
        self.clim = clim
        if self.symm:
            self.clim = self.symm_clim()
        return self.clim

    def symm_clim(self):
        """
        Make color limits symmetric around zero.

        Returns
        -------
        tuple
            Symmetric color limits (-max, max)
        """
        abs_max = np.max(np.abs(self.clim))
        return (-abs_max, abs_max)

    def get_cbar_extend(self, input, clim=None):
        """
        Determine colorbar extension mode based on data and limits.

        Parameters
        ----------
        input : array_like
            Input data array
        clim : tuple, optional
            Color limits (min, max), if None uses current or calculated limits

        Returns
        -------
        str
            Colorbar extension mode: 'neither', 'min', 'max', or 'both'
        """
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

    def get_norm(self, lognorm=False):
        """
        Get normalization for colormap.

        Parameters
        ----------
        lognorm : bool, optional
            Whether to use logarithmic normalization, default is False

        Returns
        -------
        matplotlib.colors.Normalize
            Normalization object for matplotlib colormaps

        Notes
        -----
        Requires the clim attribute to be set (by calling get_clim or find_common_clim)
        """
        vmin, vmax = self.clim
        if lognorm:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        return norm


class Raster:
    """
    Utility class for raster data handling and processing.

    This class provides convenient methods for reading, processing,
    and analyzing raster data using rasterio and GDAL.

    Attributes
    ----------
    fn : str
        Path to the raster file
    ds : rasterio.DatasetReader
        Open rasterio dataset

    Examples
    --------
    >>> raster = Raster("path/to/dem.tif")
    >>> data = raster.read_array()
    >>> hillshade = raster.hillshade()
    >>> epsg = raster.get_epsg_code()
    >>> gsd = raster.get_gsd()
    """

    def __init__(self, fn):
        """
        Initialize the Raster object.

        Parameters
        ----------
        fn : str
            Path to the raster file
        """
        self.fn = fn
        self.ds = rio.open(fn)

    def read_array(self, b=1, extent=False):
        """
        Read raster data as a numpy masked array.

        Parameters
        ----------
        b : int, optional
            Band number to read, default is 1
        extent : bool, optional
            Whether to return extent information for plotting, default is False

        Returns
        -------
        numpy.ma.MaskedArray or tuple
            If extent=False: masked array of raster data
            If extent=True: tuple of (masked array, extent)

        Notes
        -----
        No-data values are properly masked, and invalid values are fixed.
        """
        a = self.ds.read(b, masked=True)
        ndv = self.get_ndv()
        ma = np.ma.fix_invalid(np.ma.masked_equal(a, ndv))
        out = ma
        if extent:
            extent = rio.plot.plotting_extent(self.ds)
            out = (ma, extent)
        return out

    def read_raster_subset(self, bbox, b=1):
        """
        Read a subset of raster data defined by a bounding box.

        Parameters
        ----------
        bbox : tuple
            Bounding box in the format (ul_x, lr_y, lr_x, ul_y)
            (upper-left x, lower-right y, lower-right x, upper-left y)
        b : int, optional
            Band number to read, default is 1

        Returns
        -------
        numpy.ndarray
            Subset of raster data
        """
        window = from_bounds(*bbox, self.ds.transform)
        subset = self.ds.read(b, window=window)
        return subset

    def get_ndv(self):
        """
        Get the no-data value for the raster.

        Returns
        -------
        float or int
            No-data value

        Notes
        -----
        If no-data value is not defined in the raster metadata,
        tries to infer it from the first pixel value.
        """
        ndv = self.ds.nodatavals[0]
        if ndv is None:
            ndv = self.ds.read(1, window=Window(0, 0, 1, 1)).squeeze()
        return ndv

    def get_epsg_code(self):
        """
        Get the EPSG code for the raster's coordinate reference system.

        Returns
        -------
        int
            EPSG code
        """
        epsg = self.ds.crs.to_epsg()
        return epsg

    def get_gsd(self):
        """
        Get the ground sample distance (resolution) of the raster.

        Returns
        -------
        float
            Ground sample distance (pixel size) in raster units
        """
        gsd = self.ds.transform[0]
        return gsd

    def get_bounds(self, latlon=True, json_format=True):
        """
        Get the geographic bounds of the raster.

        Parameters
        ----------
        latlon : bool, optional
            Whether to return bounds in latitude/longitude, default is True
        json_format : bool, optional
            Whether to return bounds in GeoJSON-like format, default is True

        Returns
        -------
        list or tuple
            If json_format=True: list of corner coordinates as dictionaries
            If json_format=False: tuple of (min_x, min_y, max_x, max_y)
        """
        ds = rioxarray.open_rasterio(self.fn, masked=True).squeeze()
        bounds = ds.rio.bounds()
        if latlon:
            epsg = self.get_epsg_code()
            bounds = rio.warp.transform_bounds(f"EPSG:{epsg}", "EPSG:4326", *bounds)
        if json_format:
            min_lon, min_lat, max_lon, max_lat = bounds
            region = [
                {"lon": min_lon, "lat": min_lat},
                {"lon": min_lon, "lat": max_lat},
                {"lon": max_lon, "lat": max_lat},
                {"lon": max_lon, "lat": min_lat},
                {"lon": min_lon, "lat": min_lat},
            ]
            return region
        else:
            return bounds

    def hillshade(self):
        """
        Generate a hillshade from the raster.

        Returns
        -------
        numpy.ma.MaskedArray
            Hillshade array

        Notes
        -----
        First checks if a hillshade file already exists with "_hs.tif" suffix.
        If not, generates the hillshade using GDAL.
        """
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
        """
        Compute the difference between this raster and another.

        Parameters
        ----------
        second_fn : str
            Path to the second raster file

        Returns
        -------
        numpy.ndarray
            Difference array (second_raster - this_raster)

        Notes
        -----
        Uses geoutils to align rasters before differencing.
        Saves the difference raster to disk with "_diff.tif" suffix.
        """
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
    """
    Base class for plotting array and vector data.

    This class provides common plotting functionality, including color management,
    colorbar customization, and basemap addition.

    Attributes
    ----------
    clim_perc : tuple
        Percentile range for color limits, default is (2, 98)
    lognorm : bool
        Whether to use logarithmic color normalization, default is False
    title : str or None
        Plot title, default is None
    cb : ColorBar
        ColorBar instance for managing color scaling

    Examples
    --------
    >>> plotter = Plotter(clim_perc=(5, 95), title="My Plot")
    >>> fig, ax = plt.subplots()
    >>> plotter.plot_array(ax, data, cmap="viridis", cbar_label="Elevation (m)")
    """

    def __init__(
        self,
        clim_perc=(2, 98),
        lognorm=False,
        title=None,
    ):
        """
        Initialize the Plotter.

        Parameters
        ----------
        clim_perc : tuple, optional
            Percentile range for color limits, default is (2, 98)
        lognorm : bool, optional
            Whether to use logarithmic color normalization, default is False
        title : str, optional
            Plot title, default is None
        """
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
        """
        Plot a 2D array on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        array : array_like
            2D array to plot
        clim : tuple, optional
            Color limits (min, max), default is None (auto-calculated)
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use, default is "inferno"
        add_cbar : bool, optional
            Whether to add a colorbar, default is True
        cbar_label : str, optional
            Label for the colorbar, default is None
        alpha : float, optional
            Transparency (0-1), default is 1

        Returns
        -------
        matplotlib.image.AxesImage
            The plotted image

        Notes
        -----
        If clim is None, color limits are calculated using the ColorBar instance.
        """
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

        return im

    def plot_geodataframe(
        self,
        ax,
        gdf,
        column_name,
        clim=None,
        cmap="inferno",
        cbar_label=None,
        **ctx_kwargs,
    ):
        """
        Plot a GeoDataFrame with color mapping and optional basemap.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        gdf : geopandas.GeoDataFrame
            GeoDataFrame to plot
        column_name : str
            Column name to use for color mapping
        clim : tuple, optional
            Color limits (min, max), default is None (auto-calculated)
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap to use, default is "inferno"
        cbar_label : str, optional
            Label for the colorbar, default is None
        **ctx_kwargs
            Additional keyword arguments for contextily.add_basemap

        Returns
        -------
        matplotlib.axes.Axes
            The axes with the plot

        Notes
        -----
        If ctx_kwargs are provided, adds a basemap using contextily.
        """
        if clim is None:
            self.cb.get_clim(gdf[column_name])
        else:
            self.cb.clim = clim
        norm = self.cb.get_norm(self.lognorm)

        gdf.plot(
            ax=ax,
            column=column_name,
            cmap=cmap,
            norm=norm,
            s=1,
            legend=True,
            legend_kwds={"label": cbar_label},
        )

        if ctx_kwargs:
            ctx.add_basemap(ax=ax, **ctx_kwargs)
