import glob
import logging
import os
import re
from datetime import datetime

from asp_plot.utils import glob_file

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ProcessingParameters:
    def __init__(
        self, processing_directory, bundle_adjust_directory=None, stereo_directory=None
    ):
        self.processing_directory = processing_directory
        self.bundle_adjust_directory = bundle_adjust_directory
        self.stereo_directory = stereo_directory
        self.full_ba_directory = (
            os.path.join(self.processing_directory, bundle_adjust_directory)
            if bundle_adjust_directory
            else None
        )
        self.full_stereo_directory = (
            os.path.join(self.processing_directory, stereo_directory)
            if stereo_directory
            else None
        )
        self.processing_parameters_dict = {}

        try:
            self.bundle_adjust_log = glob_file(
                self.full_ba_directory, "*log-bundle_adjust*.txt"
            )
        except:
            self.bundle_adjust_log = None
        try:
            self.stereo_logs = glob.glob(
                os.path.join(self.full_stereo_directory, "*log-stereo*.txt")
            )
        except:
            self.stereo_logs = None
        try:
            self.point2dem_log = glob_file(
                self.full_stereo_directory, "*log-point2dem*.txt"
            )
        except:
            self.point2dem_log = None

    def from_log_files(self):
        if self.bundle_adjust_directory:
            bundle_adjust_params, ba_run_time, reference_dem = (
                self.from_bundle_adjust_log()
            )
            if reference_dem != "":
                processing_timestamp, stereo_params, stereo_run_time = (
                    self.from_stereo_log()
                )
            else:
                processing_timestamp, stereo_params, stereo_run_time, reference_dem = (
                    self.from_stereo_log(search_for_reference_dem=True)
                )
            bundle_adjust_params = (
                "bundle_adjust " + bundle_adjust_params.split(maxsplit=1)[1]
            )
        else:
            bundle_adjust_params = "Bundle adjustment not run"
            ba_run_time = "N/A"
            processing_timestamp, stereo_params, stereo_run_time, reference_dem = (
                self.from_stereo_log(search_for_reference_dem=True)
            )

        point2dem_params, point2dem_run_time = self.from_point2dem_log()

        stereo_params = "stereo " + stereo_params.split(maxsplit=1)[1]
        point2dem_params = "point2dem " + point2dem_params.split(maxsplit=1)[1]

        self.processing_parameters_dict = {
            "processing_timestamp": processing_timestamp,
            "reference_dem": reference_dem,
            "bundle_adjust": bundle_adjust_params,
            "bundle_adjust_run_time": ba_run_time,
            "stereo": stereo_params,
            "stereo_run_time": stereo_run_time,
            "point2dem": point2dem_params,
            "point2dem_run_time": point2dem_run_time,
        }

        return self.processing_parameters_dict

    def from_bundle_adjust_log(self):
        if not self.bundle_adjust_log:
            raise ValueError(
                f"\n\nCould not find bundle adjust log file in {self.full_ba_directory}\nCheck that the *log*.txt file exists in the directory specified.\n\n"
            )
        bundle_adjust_params = ""
        with open(self.bundle_adjust_log, "r") as file:
            for line in file:
                if "bundle_adjust" in line and not bundle_adjust_params:
                    bundle_adjust_params = line.strip()
                    break

        reference_dem = self.get_reference_dem(
            self.bundle_adjust_log, starting_string="Loading DEM:"
        )

        run_time = self.get_run_time([self.bundle_adjust_log])

        return bundle_adjust_params, run_time, reference_dem

    def from_point2dem_log(self):
        if not self.point2dem_log:
            raise ValueError(
                f"\n\nCould not find point2dem log file in {self.full_stereo_directory}\nCheck that the *log*.txt file exists in the directory specified.\n\n"
            )
        point2dem_params = ""
        with open(self.point2dem_log, "r") as file:
            for line in file:
                if "point2dem" in line and not point2dem_params:
                    point2dem_params = line.strip()
                    break

        run_time = self.get_run_time([self.point2dem_log])

        return point2dem_params, run_time

    def from_stereo_log(self, search_for_reference_dem=False):
        # Stereo proceeds as:
        #  1. stereo_pprc
        #  2. stereo_corr
        #  3. stereo_blend (logs in tile/ dirs)
        #  4. stereo_rfne (logs in tile/ dirs)
        #  5. stereo_fltr
        #  6. stereo_tri
        if not self.stereo_logs:
            raise ValueError(
                f"\n\nCould not find stereo log files in {self.full_stereo_directory}\nCheck that these *log*.txt files exist in the directory specified.\n\n"
            )
        pprc_log = next(
            (log for log in self.stereo_logs if "log-stereo_pprc" in log), None
        )
        tri_log = next(
            (log for log in self.stereo_logs if "log-stereo_tri" in log), None
        )
        stereo_params = ""
        with open(tri_log, "r") as file:
            for line in file:
                if "stereo" in line and not stereo_params:
                    stereo_params = line.strip()
                    break

        processing_timestamp = ""
        with open(pprc_log, "r") as file:
            for line in file:
                if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
                    processing_timestamp = datetime.strptime(
                        line.split()[0] + " " + line.split()[1], "%Y-%m-%d %H:%M:%S"
                    )

        run_time = self.get_run_time([pprc_log, tri_log])

        if search_for_reference_dem:
            reference_dem = self.get_reference_dem(
                pprc_log, starting_string="Using input DEM:"
            )

            return processing_timestamp, stereo_params, run_time, reference_dem
        else:
            return processing_timestamp, stereo_params, run_time

    def get_reference_dem(self, logfile, starting_string="DEM:"):
        reference_dem = ""
        with open(logfile, "r") as file:
            for line in file:
                if starting_string in line:
                    reference_dem = line.split(starting_string)[1].strip()
        return reference_dem

    def get_run_time(self, logfiles):
        start_time = None
        end_time = None

        start_log = logfiles[0]
        end_log = logfiles[-1] if len(logfiles) > 1 else start_log

        with open(start_log, "r") as file:
            for line in file:
                if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
                    start_time = datetime.strptime(
                        line.split()[0] + " " + line.split()[1], "%Y-%m-%d %H:%M:%S"
                    )
                    break

        with open(end_log, "r") as file:
            for line in file:
                if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
                    end_time = datetime.strptime(
                        line.split()[0] + " " + line.split()[1], "%Y-%m-%d %H:%M:%S"
                    )

        if start_time and end_time:
            time_diff = end_time - start_time
            hours, remainder = divmod(time_diff.total_seconds(), 3600)
            minutes = remainder // 60
            run_time = f"{int(hours)} hours and {int(minutes)} minutes"
        else:
            run_time = "N/A"

        return run_time
