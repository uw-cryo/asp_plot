import logging
import os
import re

import numpy as np
from osgeo import gdal, osr

from asp_plot.utils import Raster, glob_file, run_subprocess_command

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Alignment:
    """
    Perform DEM alignment using point cloud alignment techniques.

    This class provides functionality to align DEMs to reference datasets,
    particularly ICESat-2 ATL06SR altimetry data, using point cloud
    alignment methods. It includes methods to run the alignment process,
    extract alignment statistics, and apply the alignment transformation
    to the DEM.

    Attributes
    ----------
    directory : str
        Root directory for alignments and outputs
    dem_fn : str
        Path to the DEM file to be aligned

    Examples
    --------
    >>> alignment = Alignment('/path/to/directory', '/path/to/dem.tif')
    >>> alignment.pc_align_dem_to_atl06sr(atl06sr_csv='/path/to/icesat2_data.csv')
    >>> report = alignment.pc_align_report()
    >>> aligned_dem = alignment.apply_dem_translation()
    """

    def __init__(
        self,
        directory,
        dem_fn,
        **kwargs,
    ):
        """
        Initialize the Alignment object.

        Parameters
        ----------
        directory : str
            Root directory for alignments and outputs
        dem_fn : str
            Path to the DEM file to be aligned
        **kwargs : dict, optional
            Additional keyword arguments for future extensions

        Raises
        ------
        ValueError
            If the DEM file does not exist
        """
        self.directory = directory

        if not os.path.exists(dem_fn):
            raise ValueError(f"DEM file not found: {dem_fn}")
        self.dem_fn = dem_fn

    def pc_align_dem_to_atl06sr(
        self,
        max_displacement=20,
        max_source_points=10000000,
        alignment_method="point-to-point",
        atl06sr_csv=None,
        output_prefix="pc_align/pc_align",
    ):
        """
        Align DEM to ICESat-2 ATL06SR data using point cloud alignment.

        Runs the ASP pc_align tool to align the DEM to ICESat-2 ATL06SR
        altimetry data. The resulting transformation parameters are saved
        to log files for later application.

        Parameters
        ----------
        max_displacement : float, optional
            Maximum expected displacement in meters, default is 20
        max_source_points : int, optional
            Maximum number of source points to use, default is 10,000,000
        alignment_method : str, optional
            Method for alignment, default is "point-to-point"
        atl06sr_csv : str, optional
            Path to the ATL06SR CSV file, default is None
        output_prefix : str, optional
            Prefix for output files, default is "pc_align/pc_align"

        Raises
        ------
        ValueError
            If the ATL06SR CSV file is not provided or does not exist

        Notes
        -----
        The ATL06SR CSV file must be formatted with columns for longitude,
        latitude, and height above datum. This can be created using the
        to_csv_for_pc_align() function in the Altimetry module.
        """
        if atl06sr_csv is None or not os.path.exists(atl06sr_csv):
            raise ValueError(
                f"\nATL06 filtered CSV file not found: {atl06sr_csv}\nWe need this to run pc_align. It can be created with the to_csv_for_pc_align() function in the Altimetry module.\n"
            )

        pc_align_folder = os.path.join(self.directory, output_prefix)

        print(
            f"Running pc_align on {self.dem_fn} and {atl06sr_csv}\nWriting to {pc_align_folder}*"
        )

        command = [
            "pc_align",
            "--max-displacement",
            str(max_displacement),
            "--max-num-source-points",
            str(max_source_points),
            "--alignment-method",
            alignment_method,
            "--csv-format",
            "1:lon 2:lat 3:height_above_datum",
            "--compute-translation-only",
            "--output-prefix",
            pc_align_folder,
            self.dem_fn,
            atl06sr_csv,
        ]

        run_subprocess_command(command)

    def pc_align_report(self, output_prefix="pc_align/pc_align"):
        """
        Extract alignment statistics from pc_align log files.

        Parses the pc_align log file to extract error percentiles,
        translation vectors, and other alignment metrics.

        Parameters
        ----------
        output_prefix : str, optional
            Prefix for pc_align output files, default is "pc_align/pc_align"

        Returns
        -------
        dict
            Dictionary containing alignment metrics:
            - p16_beg, p50_beg, p84_beg: Error percentiles before alignment
            - p16_end, p50_end, p84_end: Error percentiles after alignment
            - x_shift, y_shift, z_shift: Translation vector components in ECEF
            - translation_magnitude: Magnitude of translation vector

        Notes
        -----
        This method expects the log file to contain specific keyword patterns
        that match the pc_align output format. If the log format changes,
        this parser may need to be updated.
        """
        pc_align_log = glob_file(self.directory, f"{output_prefix}-log-pc_align*.txt")

        with open(pc_align_log, "r") as file:
            content = file.readlines()

        report = {}
        for line in content:
            if "Input: error percentile" in line:
                values = re.findall(r"(?:\d+%: )(\d+\.\d+)", line)
                percentile_dict = {
                    "p16_beg": float(values[0]),
                    "p50_beg": float(values[1]),
                    "p84_beg": float(values[2]),
                }
                report = report | percentile_dict
            if "Output: error percentile" in line:
                values = re.findall(r"(?:\d+%: )(\d+\.\d+)", line)
                percentile_dict = {
                    "p16_end": float(values[0]),
                    "p50_end": float(values[1]),
                    "p84_end": float(values[2]),
                }
                report = report | percentile_dict
            if "Translation vector (Cartesian, meters):" in line:
                ecef_shift = np.genfromtxt(
                    [line.split("Vector3")[1][1:-2]], delimiter=","
                )
                xyz_shift_dict = {
                    "x_shift": ecef_shift[0],
                    "y_shift": ecef_shift[1],
                    "z_shift": ecef_shift[2],
                }
                report = report | xyz_shift_dict
            if "Translation vector magnitude (meters):" in line:
                magnitude = re.findall(r"magnitude \(meters\): (\d+\.\d+)", line)[0]
                report["translation_magnitude"] = float(magnitude)

        return report

    def apply_dem_translation(self, output_prefix="pc_align/pc_align"):
        """
        Apply the pc_align translation to the DEM.

        Creates a translated version of the DEM by applying the transformation
        parameters extracted from pc_align log files. This method directly
        modifies the DEM's geotransform and pixel values without resampling.

        Parameters
        ----------
        output_prefix : str, optional
            Prefix for pc_align output files, default is "pc_align/pc_align"

        Returns
        -------
        str
            Path to the translated DEM file

        Raises
        ------
        ValueError
            If the log file does not contain necessary translation information

        Notes
        -----
        This method is faster and more precise than using point2dem
        to generate a new DEM from translated point cloud, as it directly
        applies the translation to the DEM's metadata and pixel values.
        """
        pc_align_log = glob_file(self.directory, f"{output_prefix}-log-pc_align*.txt")

        src = Raster(self.dem_fn)
        src_a = src.read_array()
        src_ndv = src.get_ndv()

        # Need to extract from log to know how to compute translation
        # if ref is csv and src is dem, want to transform source_center + shift
        # if ref is dem and src is csv, want to inverse transform ref by shift applied at (source_center - shift)

        llz_c = None
        with open(pc_align_log, "r") as file:
            content = file.readlines()

        for line in content:
            if "Centroid of source points (Cartesian, meters):" in line:
                ecef_c = np.genfromtxt([line.split("Vector3")[1][1:-2]], delimiter=",")
            if "Centroid of source points (lat,lon,z):" in line:
                llz_c = np.genfromtxt([line.split("Vector3")[1][1:-2]], delimiter=",")
            if "Translation vector (Cartesian, meters):" in line:
                ecef_shift = np.genfromtxt(
                    [line.split("Vector3")[1][1:-2]], delimiter=","
                )
            if "Translation vector (lat,lon,z):" in line:
                llz_shift = np.genfromtxt(
                    [line.split("Vector3")[1][1:-2]], delimiter=","
                )
                break

        if llz_c is None:
            raise ValueError(
                f"\nLog file does not contain necessary translation information: {pc_align_log}\n"
            )

        # Reorder lat,lon,z to lon,lat,z (x,y,z)
        i = [1, 0, 2]
        llz_c = llz_c[i]
        llz_shift = llz_shift[i]

        ecef_srs = osr.SpatialReference()
        ecef_srs.ImportFromEPSG(4978)

        s_srs = ecef_srs
        src_c = ecef_c
        src_shift = ecef_shift

        # Determine shift in original dataset coords
        t_srs = osr.SpatialReference()
        t_srs.ImportFromWkt(src.ds.crs.to_wkt())
        proj_shift = self.get_proj_shift(src_c, src_shift, s_srs, t_srs, inv_trans=True)

        aligned_dem_fn = self.dem_fn.replace(".tif", "_pc_align_translated.tif")
        print(f"\nWriting out: {aligned_dem_fn}\n")

        gdal_opt = ["COMPRESS=LZW", "TILED=YES", "PREDICTOR=3", "BIGTIFF=IF_SAFER"]
        dst_ds = gdal.GetDriverByName("GTiff").CreateCopy(
            aligned_dem_fn, gdal.Open(self.dem_fn), strict=0, options=gdal_opt
        )
        # Apply vertical shift
        dst_b = dst_ds.GetRasterBand(1)
        dst_b.SetNoDataValue(float(src_ndv))
        dst_b.WriteArray(np.around((src_a + proj_shift[2]).filled(src_ndv), decimals=3))

        dst_gt = list(dst_ds.GetGeoTransform())
        # Apply horizontal shift directly to geotransform
        dst_gt[0] += proj_shift[0]
        dst_gt[3] += proj_shift[1]
        dst_ds.SetGeoTransform(dst_gt)
        dst_ds = None

        return aligned_dem_fn

    def get_proj_shift(self, src_c, src_shift, s_srs, t_srs, inv_trans=True):
        """
        Calculate projection shift between coordinate systems.

        Transforms a shift vector from one coordinate system to another,
        accounting for the non-linearity of coordinate transformations.

        Parameters
        ----------
        src_c : numpy.ndarray
            Source point coordinates in source coordinate system
        src_shift : numpy.ndarray
            Shift vector in source coordinate system
        s_srs : osr.SpatialReference
            Source spatial reference system
        t_srs : osr.SpatialReference
            Target spatial reference system
        inv_trans : bool, optional
            Whether to invert the transformation direction, default is True

        Returns
        -------
        numpy.ndarray
            Shift vector in target coordinate system

        Notes
        -----
        This is a helper method used by apply_dem_translation to convert
        shifts between coordinate systems. The method handles both same-system
        transformations and cross-system transformations.
        """
        if s_srs.IsSame(t_srs):
            proj_shift = src_shift
        else:
            src_c_shift = src_c + src_shift
            src2proj = osr.CoordinateTransformation(s_srs, t_srs)
            proj_c = np.array(src2proj.TransformPoint(*src_c))
            proj_c_shift = np.array(src2proj.TransformPoint(*src_c_shift))
            if inv_trans:
                proj_shift = proj_c - proj_c_shift
            else:
                proj_shift = proj_c_shift - proj_c
        # Reduce unnecessary precision
        proj_shift = np.around(proj_shift, 3)
        return proj_shift

    # DEPRECATED: Very slow, better to use apply_dem_translation().
    # def generate_translated_dem(self, pc_align_output, dem_out_fn):
    #     """
    #     Very slow, better to use apply_dem_translation().
    #     """
    #     if not os.path.exists(pc_align_output):
    #         raise ValueError(
    #             f"\npc_align output not found: {pc_align_output}\n\nWe need this to generate the translated DEM.\n"
    #         )

    #     rast = Raster(self.dem_fn)
    #     gsd = rast.get_gsd()
    #     epsg = rast.get_epsg_code()

    #     command = [
    #         "point2dem",
    #         "--tr",
    #         str(gsd),
    #         "--t_srs",
    #         f"EPSG:{epsg}",
    #         "--nodata-value",
    #         str(-9999),
    #         "-o",
    #         dem_out_fn,
    #         pc_align_output,
    #     ]

    #     run_subprocess_command(command)
