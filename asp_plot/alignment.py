import logging
import os

import numpy as np
from osgeo import gdal, osr

from asp_plot.utils import Raster, glob_file, run_subprocess_command

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class Alignment:
    def __init__(
        self,
        directory,
        dem_fn,
        # aligned_dem_fn=None,
        **kwargs,
    ):
        self.directory = directory

        if not os.path.exists(dem_fn):
            raise ValueError(f"DEM file not found: {dem_fn}")
        self.dem_fn = dem_fn

        # if aligned_dem_fn is not None and not os.path.exists(aligned_dem_fn):
        #     raise ValueError(f"Aligned DEM file not found: {aligned_dem_fn}")
        # self.aligned_dem_fn = aligned_dem_fn

    # TODO: move all pc_align steps and dem translations to separate class
    # call this alignment for different processing levels, with some minimum number
    # of required points (a ~500 point parameter).
    # Report all translation results to user via a dictionary and printing.
    # If translation agrees within some tolerance (a 5% or 10% parameter) in XYZ components,
    # apply the translation of the points found closest in time to the DEM.
    def pc_align_dem_to_atl06sr(
        self,
        max_displacement=20,
        max_source_points=10000000,
        alignment_method="point-to-point",
        atl06sr_csv=None,
        output_prefix="pc_align/pc_align",
    ):
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
        pc_align_log = glob_file(self.directory, f"{output_prefix}-log-pc_align*.txt")

        with open(pc_align_log, "r") as file:
            content = file.readlines()

        report = ""
        for line in content:
            if "Input: error percentile of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Input: mean of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Output: error percentile of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Output: mean of smallest errors (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
            if "Translation vector (Cartesian, meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line
                ecef_shift = np.genfromtxt(
                    [line.split("Vector3")[1][1:-2]], delimiter=","
                )
                report += f"\nECEF shift: {ecef_shift}\n"
            if "Translation vector magnitude (meters):" in line:
                report_line = line.split("[ console ] : ")[1]
                report += report_line

        return report

    def apply_dem_translation(self, output_prefix="pc_align/pc_align", inv_trans=True):
        def get_proj_shift(src_c, src_shift, s_srs, t_srs, inv_trans=True):
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
        proj_shift = get_proj_shift(src_c, src_shift, s_srs, t_srs, inv_trans)

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
