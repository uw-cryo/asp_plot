import os
import glob
import matplotlib.pyplot as plt
from asp_plot.utils import save_figure


class ProcessingParameters:
    def __init__(self, directory, bundle_adjust_directory, stereo_directory):
        self.directory = directory
        self.bundle_adjust_directory = bundle_adjust_directory
        self.stereo_directory = stereo_directory
        self.processing_parameters_dict = {}

        try:
            self.bundle_adjust_log = glob.glob(
                os.path.join(self.directory, self.bundle_adjust_directory, "*log*.txt")
            )[0]
            self.stereo_log = glob.glob(
                os.path.join(self.directory, self.stereo_directory, "*log-stereo*.txt")
            )[0]
            self.point2dem_log = glob.glob(
                os.path.join(
                    self.directory, self.stereo_directory, "*log-point2dem*.txt"
                )
            )[0]
        except:
            raise ValueError(
                "Could not find log files in bundle adjust and stereo directories\nCheck that these *log*.txt files exist in the directories specified"
            )

    def from_log_files(self):
        with open(self.bundle_adjust_log, "r") as file:
            content = file.readlines()

        bundle_adjust_params = ""
        processing_timestamp = ""

        for line in content:
            if "bundle_adjust" in line and not bundle_adjust_params:
                bundle_adjust_params = line.strip()

            if "[ console ]" in line and not processing_timestamp:
                date, time = line.split()[0], line.split()[1]
                processing_timestamp = f"{date}-{time[:5].replace(':', '')}"

            if bundle_adjust_params and processing_timestamp:
                break

        with open(self.stereo_log, "r") as file:
            content = file.readlines()

        stereo_params = ""

        for line in content:
            if "stereo" in line and not stereo_params:
                stereo_params = line.strip()

            if stereo_params:
                break

        with open(self.point2dem_log, "r") as file:
            content = file.readlines()

        point2dem_params = ""

        for line in content:
            if "point2dem" in line and not point2dem_params:
                point2dem_params = line.strip()

            if point2dem_params:
                break

        self.processing_parameters_dict = {
            "bundle_adjust": bundle_adjust_params,
            "stereo": stereo_params,
            "point2dem": point2dem_params,
            "processing_timestamp": processing_timestamp,
        }

        return self.processing_parameters_dict

    def plot_processing_parameters(self, save_dir=None, fig_fn=None):
        fig, axes = plt.subplots(3, 1, figsize=(10, 6))
        ax1, ax2, ax3 = axes.flatten()

        ax1.axis("off")
        ax1.text(
            0.5,
            0.5,
            f"Processed on: {self.processing_parameters_dict['processing_timestamp']:}\n\nBundle Adjust:\n{self.processing_parameters_dict['bundle_adjust']:}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            wrap=True,
        )

        ax2.axis("off")
        ax2.text(
            0.5,
            0.5,
            f"Stereo:\n{self.processing_parameters_dict['stereo']:}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            wrap=True,
        )

        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            f"point2dem:\n{self.processing_parameters_dict['point2dem']:}",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            wrap=True,
        )

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
