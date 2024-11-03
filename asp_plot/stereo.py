import logging
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib_scalebar.scalebar import ScaleBar

from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.utils import ColorBar, Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class StereoPlotter(Plotter):
    def __init__(
        self, directory, stereo_directory, reference_dem=None, out_dem_gsd=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.directory = directory
        self.stereo_directory = stereo_directory

        if reference_dem:
            self.reference_dem = reference_dem
        else:
            processing_parameters = ProcessingParameters(
                processing_directory=self.directory,
                stereo_directory=self.stereo_directory,
            )
            self.reference_dem = processing_parameters.from_stereo_log(
                search_for_reference_dem=True
            )[-1]
            if not self.reference_dem:
                logger.warning(
                    "\n\nNo reference DEM found in log files. Please supply the reference DEM you used during stereo processing (or another reference DEM) if you would like to see some difference maps.\n\n"
                )
        if self.reference_dem:
            print(f"\nReference DEM: {self.reference_dem}\n")

        self.out_dem_gsd = out_dem_gsd

        self.full_directory = os.path.join(self.directory, self.stereo_directory)
        self.left_ortho_sub_fn = glob_file(self.full_directory, "*-L_sub.tif")
        self.right_ortho_sub_fn = glob_file(self.full_directory, "*-R_sub.tif")
        self.left_ortho_fn = glob_file(self.full_directory, "*-L.tif")
        self.match_point_fn = glob_file(self.full_directory, "*.match")
        self.disparity_sub_fn = glob_file(self.full_directory, "*-D_sub.tif")
        self.disparity_fn = glob_file(self.full_directory, "*-D.tif")

        self.dem_fn = glob_file(
            self.full_directory,
            f"*-DEM_{self.out_dem_gsd}m.tif",
            f"*{self.out_dem_gsd}m-DEM.tif",
        )
        if not self.dem_fn:
            raise ValueError(
                f"\n\nNo DEM found in {self.full_directory} with GSD {self.out_dem_gsd} m. Please run stereo processing with the desired output DEM GSD or correct your inputs here.\n\n"
            )
        else:
            print(f"\nASP DEM: {self.dem_fn}\n")

        self.intersection_error_fn = glob_file(
            self.full_directory, "*-IntersectionErr.tif"
        )

    def read_ip_record(self, match_file):
        x, y = np.frombuffer(match_file.read(8), dtype=np.float32)
        xi, yi = np.frombuffer(match_file.read(8), dtype=np.int32)
        orientation, scale, interest = np.frombuffer(
            match_file.read(12), dtype=np.float32
        )
        (polarity,) = np.frombuffer(match_file.read(1), dtype=bool)
        octave, scale_lvl = np.frombuffer(match_file.read(8), dtype=np.uint32)
        (ndesc,) = np.frombuffer(match_file.read(8), dtype=np.uint64)
        desc = np.frombuffer(match_file.read(int(ndesc * 4)), dtype=np.float32)
        iprec = [
            x,
            y,
            xi,
            yi,
            orientation,
            scale,
            interest,
            polarity,
            octave,
            scale_lvl,
            ndesc,
        ]
        iprec.extend(desc)
        return iprec

    def get_match_point_df(self):
        out_csv = (
            os.path.splitext(self.match_point_fn)[0] + ".csv"
            if self.match_point_fn
            else None
        )
        if self.match_point_fn and not os.path.exists(out_csv):
            with (
                open(self.match_point_fn, "rb") as match_file,
                open(out_csv, "w") as out,
            ):
                size1 = np.frombuffer(match_file.read(8), dtype=np.uint64)[0]
                size2 = np.frombuffer(match_file.read(8), dtype=np.uint64)[0]
                out.write("x1 y1 x2 y2\n")
                im1_ip = [self.read_ip_record(match_file) for i in range(size1)]
                im2_ip = [self.read_ip_record(match_file) for i in range(size2)]
                for i in range(len(im1_ip)):
                    out.write(
                        "{} {} {} {}\n".format(
                            im1_ip[i][0], im1_ip[i][1], im2_ip[i][0], im2_ip[i][1]
                        )
                    )

        return (
            pd.read_csv(out_csv, delimiter=r"\s+")
            if out_csv and os.path.exists(out_csv)
            else None
        )

    def plot_match_points(self, save_dir=None, fig_fn=None):
        match_point_df = self.get_match_point_df()

        fig, axa = plt.subplots(1, 2, figsize=(10, 5))

        if (
            self.left_ortho_sub_fn
            and self.right_ortho_sub_fn
            and match_point_df is not None
        ):
            full_gsd = Raster(self.left_ortho_fn).get_gsd()
            sub_gsd = Raster(self.left_ortho_sub_fn).get_gsd()
            rescale_factor = sub_gsd / full_gsd

            left_image = Raster(self.left_ortho_sub_fn).read_array()
            right_image = Raster(self.right_ortho_sub_fn).read_array()

            self.plot_array(ax=axa[0], array=left_image, cmap="gray", add_cbar=False)
            axa[0].set_title(f"Left image (n={match_point_df.shape[0]})")
            self.plot_array(ax=axa[1], array=right_image, cmap="gray", add_cbar=False)
            axa[1].set_title("Right image")

            axa[0].scatter(
                match_point_df["x1"] / rescale_factor,
                match_point_df["y1"] / rescale_factor,
                color="r",
                marker="o",
                facecolor="none",
                s=1,
            )
            axa[0].set_aspect("equal")

            axa[1].scatter(
                match_point_df["x2"] / rescale_factor,
                match_point_df["y2"] / rescale_factor,
                color="r",
                marker="o",
                facecolor="none",
                s=1,
            )
            axa[1].set_aspect("equal")
        else:
            axa[0].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[0].transAxes,
            )
            axa[1].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[1].transAxes,
            )

        fig.suptitle(self.title, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_disparity(
        self, unit="pixels", remove_bias=True, quiver=True, save_dir=None, fig_fn=None
    ):
        if unit not in ["pixels", "meters"]:
            raise ValueError(
                "\n\nUnit for disparity plot must be either 'pixels' or 'meters'\n\n"
            )

        fig, axa = plt.subplots(1, 3, figsize=(10, 3), dpi=220)
        fig.suptitle(self.title, size=10)

        if self.disparity_sub_fn and self.disparity_fn:
            raster = Raster(self.disparity_sub_fn)
            sub_gsd = raster.get_gsd()
            dx = raster.read_array(b=1)
            dy = raster.read_array(b=2)

            full_gsd = Raster(self.disparity_fn).get_gsd()
            rescale_factor = sub_gsd / full_gsd
            dx *= rescale_factor
            dy *= rescale_factor

            if unit == "meters":
                dx *= full_gsd
                dy *= full_gsd

            if remove_bias:
                dx_offset = np.ma.median(dx)
                dy_offset = np.ma.median(dy)
                dx -= dx_offset
                dy -= dy_offset

            dm = np.sqrt(abs(dx**2 + dy**2))
            clim = ColorBar(symm=True).get_clim(dm)

            self.plot_array(
                ax=axa[0], array=dx, cmap="RdBu", clim=clim, cbar_label=unit
            )
            self.plot_array(
                ax=axa[1], array=dy, cmap="RdBu", clim=clim, cbar_label=unit
            )
            self.plot_array(
                ax=axa[2], array=dm, cmap="inferno", clim=(0, clim[1]), cbar_label=unit
            )

            if quiver:
                fraction = 30
                stride = max(1, int(dx.shape[1] / fraction))
                iy, ix = np.indices(dx.shape)[:, ::stride, ::stride]
                dx_q = dx[::stride, ::stride]
                dy_q = dy[::stride, ::stride]
                axa[2].quiver(ix, iy, dx_q, dy_q, color="white")

            scalebar = ScaleBar(sub_gsd)
            axa[0].add_artist(scalebar)
            axa[0].set_title("x offset")
            axa[1].set_title("y offset")
            axa[2].set_title("offset magnitude")
        else:
            for ax in axa:
                ax.text(
                    0.5,
                    0.5,
                    "One or more required\nfiles are missing",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_detailed_hillshade(
        self,
        intersection_error_percentiles=[16, 50, 84],
        subset_km=1,
        save_dir=None,
        fig_fn=None,
    ):
        # Set up the plot
        fig = plt.figure(figsize=(10, 15), dpi=220)
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1])

        # Create the large top plot
        ax_top = fig.add_subplot(gs[0, :])

        # Create the three smaller bottom plots for hillshade
        ax_hs_left = fig.add_subplot(gs[1, 0])
        ax_hs_middle = fig.add_subplot(gs[1, 1])
        ax_hs_right = fig.add_subplot(gs[1, 2])

        # Create the three smaller bottom plots for image
        ax_image_left = fig.add_subplot(gs[2, 0])
        ax_image_middle = fig.add_subplot(gs[2, 1])
        ax_image_right = fig.add_subplot(gs[2, 2])

        # Get the data
        raster = Raster(self.dem_fn)
        dem = raster.read_array()
        gsd = raster.get_gsd()
        hs = raster.hillshade()
        ie = Raster(self.intersection_error_fn).read_array()
        image = Raster(self.left_ortho_fn)

        # Full hillshade with DEM overlay
        self.plot_array(ax=ax_top, array=hs, cmap="gray", add_cbar=False)
        self.plot_array(
            ax=ax_top,
            array=dem,
            cmap="viridis",
            cbar_label="Elevation (m HAE)",
            alpha=0.5,
        )
        ax_top.set_title(self.title, size=14)
        scalebar = ScaleBar(gsd)
        ax_top.add_artist(scalebar)

        # Calculate subset size in pixels
        subset_size = int(subset_km * 1000 / gsd)

        # Calculate the number of full windows in each dimension
        rows, cols = ie.shape
        n_rows = rows // subset_size
        n_cols = cols // subset_size

        # Trim the array to fit an integer number of windows
        ie_trimmed = ie[: n_rows * subset_size, : n_cols * subset_size]

        # Reshape the array into non-overlapping blocks
        blocks = ie_trimmed.reshape(n_rows, subset_size, n_cols, subset_size).swapaxes(
            1, 2
        )

        # Calculate the percentage of valid pixels in each block
        valid_percentage = np.ma.count(blocks, axis=(2, 3)) / (
            subset_size * subset_size
        )

        # Calculate variance for each block, only for blocks with >= 90% valid pixels
        block_variances = np.ma.masked_array(
            np.zeros((n_rows, n_cols)), mask=valid_percentage < 0.9
        )

        for i in range(n_rows):
            for j in range(n_cols):
                if valid_percentage[i, j] >= 0.9:
                    block_variances[i, j] = np.ma.var(blocks[i, j])

        # Use the compressed array to calculate percentiles
        compressed_variances = block_variances.compressed()

        # Calculate the percentiles
        lower = np.percentile(compressed_variances, intersection_error_percentiles[0])
        middle = np.percentile(compressed_variances, intersection_error_percentiles[1])
        upper = np.percentile(compressed_variances, intersection_error_percentiles[2])

        # Find the indices of the blocks closest to these percentiles
        lower_idx = np.unravel_index(
            np.argmin(np.abs(block_variances - lower)), block_variances.shape
        )
        middle_idx = np.unravel_index(
            np.argmin(np.abs(block_variances - middle)), block_variances.shape
        )
        upper_idx = np.unravel_index(
            np.argmin(np.abs(block_variances - upper)), block_variances.shape
        )

        # Define distinct colors for the rectangles and subplot axes
        rect_colors = ["magenta", "cyan", "orange"]

        # Add colored boxes outlining the three areas
        percentiles_idx = [lower_idx, middle_idx, upper_idx]
        for idx, color in zip(percentiles_idx, rect_colors):
            rect = plt.Rectangle(
                (idx[1] * subset_size, idx[0] * subset_size),
                subset_size,
                subset_size,
                fill=False,
                edgecolor=color,
                linewidth=4,
            )
            ax_top.add_patch(rect)

        # Plot subsets
        axes_hillshade = [ax_hs_left, ax_hs_middle, ax_hs_right]
        axes_image = [ax_image_left, ax_image_middle, ax_image_right]
        for ax_hs, ax_img, idx, color in zip(
            axes_hillshade, axes_image, percentiles_idx, rect_colors
        ):
            hs_subset = hs[
                idx[0] * subset_size : (idx[0] + 1) * subset_size,
                idx[1] * subset_size : (idx[1] + 1) * subset_size,
            ]
            dem_subset = dem[
                idx[0] * subset_size : (idx[0] + 1) * subset_size,
                idx[1] * subset_size : (idx[1] + 1) * subset_size,
            ]
            self.plot_array(ax=ax_hs, array=hs_subset, cmap="gray", add_cbar=False)
            self.plot_array(
                ax=ax_hs,
                array=dem_subset,
                cmap="viridis",
                cbar_label="Elevation (m HAE)",
                alpha=0.5,
            )

            with rio.open(self.dem_fn) as src:
                transform = src.transform
                ul_x, ul_y = rio.transform.xy(
                    transform, idx[0] * subset_size, idx[1] * subset_size
                )
                lr_x, lr_y = rio.transform.xy(
                    transform, (idx[0] + 1) * subset_size, (idx[1] + 1) * subset_size
                )

            image_subset = image.read_raster_subset((ul_x, lr_y, lr_x, ul_y))
            clim = [image_subset.min(), np.percentile(image_subset, 95)]
            self.plot_array(
                ax=ax_img, array=image_subset, clim=clim, cmap="gray", add_cbar=False
            )

            for ax in [ax_hs, ax_img]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(None)
                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(4)

            scalebar = ScaleBar(gsd)
            ax_hs.add_artist(scalebar)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def get_diff_vs_reference(self):
        diff_fn = glob_file(self.full_directory, "*DEM*diff.tif")
        if diff_fn:
            logger.warning(
                f"\n\nFound a DEM of difference: {diff_fn}.\nUsing that for difference map plotting.\n\n"
            )
            diff = Raster(diff_fn).read_array()
        else:
            if not self.reference_dem:
                return None
            else:
                logger.warning(
                    f"\n\nNo DEM of difference found. Generating now using reference DEM: {self.reference_dem}.\n\n"
                )
                diff = Raster(self.dem_fn).compute_difference(self.reference_dem)
        return diff

    def plot_dem_results(
        self, el_clim=None, ie_clim=None, diff_clim=None, save_dir=None, fig_fn=None
    ):
        print("Plotting DEM results. This can take a minute for large inputs.")
        fig, axa = plt.subplots(1, 3, figsize=(10, 3), dpi=220)
        fig.suptitle(self.title, size=10)

        if self.dem_fn:
            raster = Raster(self.dem_fn)
            dem = raster.read_array()
            gsd = raster.get_gsd()

            hs = raster.hillshade()
            self.plot_array(ax=axa[0], array=hs, cmap="gray", add_cbar=False)
            self.plot_array(
                ax=axa[0],
                array=dem,
                clim=el_clim,
                cmap="viridis",
                cbar_label="Elevation (m HAE)",
                alpha=0.5,
            )

            scalebar = ScaleBar(gsd)
            axa[0].add_artist(scalebar)
        else:
            axa[0].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[0].transAxes,
            )
        axa[0].set_title("Stereo DEM")

        if self.intersection_error_fn:
            ie = Raster(self.intersection_error_fn).read_array()
            self.plot_array(
                ax=axa[1],
                array=ie,
                clim=ie_clim,
                cmap="inferno",
                cbar_label="Distance (m)",
            )
        else:
            axa[1].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[1].transAxes,
            )
        axa[1].set_title("Triangulation intersection error")

        diff = self.get_diff_vs_reference()
        if diff is not None:
            if diff_clim is None:
                diff_clim = ColorBar(perc_range=(2, 98), symm=True).get_clim(diff)

            self.plot_array(
                ax=axa[2],
                array=diff,
                clim=diff_clim,
                cmap="RdBu",
                cbar_label="Elevation diff. (m)",
            )
        else:
            axa[2].text(
                0.5,
                0.5,
                "One or more required\nfiles are missing",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axa[2].transAxes,
            )
        axa[2].set_title("Difference with reference DEM")

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
