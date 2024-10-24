import glob
import logging
import os
import re
from datetime import datetime

from asp_plot.utils import glob_file

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ProcessingParameters:
    def __init__(self, directory, bundle_adjust_directory, stereo_directory):
        self.directory = directory
        self.bundle_adjust_directory = bundle_adjust_directory
        self.stereo_directory = stereo_directory
        self.full_ba_directory = os.path.join(directory, bundle_adjust_directory)
        self.full_stereo_directory = os.path.join(directory, stereo_directory)
        self.processing_parameters_dict = {}

        try:
            self.bundle_adjust_log = glob_file(
                self.full_ba_directory, "*log-bundle_adjust*.txt"
            )
            self.stereo_logs = glob.glob(
                os.path.join(self.full_stereo_directory, "*log-stereo*.txt")
            )
            self.point2dem_log = glob_file(
                self.full_stereo_directory, "*log-point2dem*.txt"
            )
        except:
            raise ValueError(
                "\n\nCould not find log files in bundle adjust and stereo directories\nCheck that these *log*.txt files exist in the directories specified.\n\n"
            )

    def from_log_files(self):
        bundle_adjust_params, processing_timestamp, ba_run_time, reference_dem = (
            self.from_bundle_adjust_log()
        )
        if reference_dem != "":
            stereo_params, stereo_run_time = self.from_stereo_log()
        else:
            stereo_params, stereo_run_time, reference_dem = self.from_stereo_log(
                search_for_reference_dem=True
            )
        point2dem_params, point2dem_run_time = self.from_point2dem_log()

        bundle_adjust_params = (
            "bundle_adjust " + bundle_adjust_params.split(maxsplit=1)[1]
        )
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
        bundle_adjust_params = ""
        with open(self.bundle_adjust_log, "r") as file:
            for line in file:
                if "bundle_adjust" in line and not bundle_adjust_params:
                    bundle_adjust_params = line.strip()
                    break

        processing_timestamp = ""
        reference_dem = ""
        with open(self.bundle_adjust_log, "r") as file:
            for line in file:
                if re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line):
                    processing_timestamp = datetime.strptime(
                        line.split()[0] + " " + line.split()[1], "%Y-%m-%d %H:%M:%S"
                    )
                if "Loading DEM:" in line:
                    reference_dem = line.split("Loading DEM:")[1].strip()

        run_time = self.get_run_time([self.bundle_adjust_log])

        return bundle_adjust_params, processing_timestamp, run_time, reference_dem

    def from_point2dem_log(self):
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

        run_time = self.get_run_time([pprc_log, tri_log])

        if search_for_reference_dem:
            reference_dem = ""
            with open(pprc_log, "r") as file:
                for line in file:
                    if "Using input DEM:" in line:
                        reference_dem = line.split("Using input DEM:")[1].strip()

            return stereo_params, run_time, reference_dem
        else:
            return stereo_params, run_time

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
