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
    """
    Visualize and analyze stereo processing results from ASP.

    This class provides methods for plotting and analyzing various outputs
    from ASP stereo processing, including DEMs, disparity maps, match points,
    and difference maps with reference DEMs.

    Attributes
    ----------
    directory : str
        Root directory for ASP processing
    stereo_directory : str
        Directory containing stereo processing outputs
    dem_gsd : float or None
        Ground sample distance of the DEM in meters
    dem_fn : str or None
        Path to the DEM file
    reference_dem : str or None
        Path to the reference DEM file
    dem : numpy.ma.MaskedArray or None
        DEM data, loaded when needed
    dem_extent : tuple or None
        Extent of the DEM for plotting
    dem_hs : numpy.ma.MaskedArray or None
        Hillshade of the DEM, generated when needed
    ref_dem : numpy.ma.MaskedArray or None
        Reference DEM data, loaded when needed
    ref_dem_extent : tuple or None
        Extent of the reference DEM for plotting
    dem_diff : numpy.ma.MaskedArray or None
        Difference between DEM and reference DEM

    Examples
    --------
    >>> plotter = StereoPlotter('/path/to/asp', 'stereo', reference_dem='ref_dem.tif')
    >>> plotter.plot_dem_results(save_dir='plots', fig_fn='dem_results.png')
    >>> plotter.plot_disparity(save_dir='plots', fig_fn='disparity.png')
    """

    def __init__(
        self,
        directory,
        stereo_directory,
        dem_gsd=None,
        dem_fn=None,
        reference_dem=None,
        **kwargs,
    ):
        """
        Initialize the StereoPlotter.

        Parameters
        ----------
        directory : str
            Root directory for ASP processing
        stereo_directory : str
            Directory containing stereo processing outputs
        dem_gsd : float, optional
            Ground sample distance of the DEM in meters
        dem_fn : str, optional
            Path to the DEM file, default is None (automatically detected)
        reference_dem : str, optional
            Path to the reference DEM file, default is None (no reference)
        **kwargs
            Additional keyword arguments passed to Plotter

        Notes
        -----
        If dem_fn is not provided, the class will try to find a DEM in the
        stereo directory with a pattern like *-DEM.tif or *_dem.tif.
        """
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

        self.full_directory = os.path.join(self.directory, self.stereo_directory)
        self.left_image_fn = glob_file(self.full_directory, "*-L.tif")
        # Set processing flag if the left image is not mapprojected
        self.orthos = False if Raster(self.left_image_fn).transform is None else True
        self.left_image_sub_fn = glob_file(self.full_directory, "*-L_sub.tif")
        self.right_image_sub_fn = glob_file(self.full_directory, "*-R_sub.tif")

        # There may be multiple match files if stereo was run with --num-matches-from-disparity.
        # In that case, filter out the match file with `-disp-` in filename.
        match_files = glob_file(self.full_directory, "*.match", all_files=True)
        self.match_point_fn = [f for f in match_files if "-disp-" not in f][0]

        self.disparity_sub_fn = glob_file(self.full_directory, "*-D_sub.tif")
        # We only need the full disparity file to retrieve the GSD for plotting
        # and rescaling below.
        self.disparity_fn = glob_file(self.full_directory, "*-D.tif")

        self.dem_gsd = dem_gsd

        if not dem_fn:
            if self.dem_gsd is not None:
                self.dem_fn = glob_file(
                    self.full_directory,
                    f"*-DEM_{self.dem_gsd}m.tif",
                    f"*{self.dem_gsd}m-DEM.tif",
                )
            else:
                self.dem_fn = glob_file(
                    self.full_directory,
                    "*-DEM.tif",
                )
        else:
            self.dem_fn = glob_file(self.full_directory, dem_fn)

        if not self.dem_fn:
            raise ValueError(
                f"\n\nDEM file not found in {self.full_directory}. Make sure it is there and possibly specify the GSD with the dem_gsd argument.\n\n"
            )
        else:
            print(f"\nASP DEM: {self.dem_fn}\n")

        self.intersection_error_fn = glob_file(
            self.full_directory, "*-IntersectionErr.tif"
        )

    def read_ip_record(self, match_file):
        """
        Read an interest point record from a binary match file.

        Parameters
        ----------
        match_file : file object
            Open binary match file positioned at the start of an interest point record

        Returns
        -------
        list
            List containing the interest point record fields, including:
            - x, y: Floating point coordinates
            - xi, yi: Integer coordinates
            - orientation, scale, interest: Feature descriptors
            - polarity: Boolean flag
            - octave, scale_lvl: Feature scale information
            - ndesc: Number of descriptor values
            - desc: Feature descriptor values

        Notes
        -----
        This method is used to parse the binary ASP match file format,
        which contains interest points from both images in a stereo pair.
        """
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
        """
        Convert a binary match file to a DataFrame of match points.

        Reads the binary match file produced by ASP stereo processing
        and converts it to a DataFrame containing matched interest points
        from the left and right images.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame with columns 'x1', 'y1', 'x2', 'y2' representing
            the coordinates of matched points in the left and right images,
            or None if the match file doesn't exist

        Notes
        -----
        This method converts the binary match file to a CSV file with the
        same base name but '.csv' extension, then reads that CSV file into
        a DataFrame. If the CSV file already exists, it is read directly.
        """
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
        """
        Plot match points between the left and right images.

        Creates a figure with two subplots showing the left and right
        orthoimages (if they are map-projected) with match points overlaid.

        Parameters
        ----------
        save_dir : str or None, optional
            Directory to save the figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for the saved figure, default is None

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        If the images are not map-projected, only the match points are shown,
        not the underlying images. The match points are displayed as small red
        circles on both images.
        """
        match_point_df = self.get_match_point_df()

        fig, axa = plt.subplots(1, 2, figsize=(10, 5))

        if (
            self.left_image_sub_fn
            and self.right_image_sub_fn
            and match_point_df is not None
        ):
            # If the images are not mapprojected, we only show the distribution of match points
            # and not the underlying images, which are rotated and difficult to plot.
            # We can revisit plotting the non-mapprojected images later, but it is challenging,
            # and likely not worthwhile, as the distribution of match points is what we are interested in.
            if self.orthos:
                full_gsd = Raster(self.left_image_fn).get_gsd()
                sub_gsd = Raster(self.left_image_sub_fn).get_gsd()
                rescale_factor = sub_gsd / full_gsd
                left_image = Raster(self.left_image_sub_fn).read_array()
                right_image = Raster(self.right_image_sub_fn).read_array()
            else:
                # These are small hacks to make the match point plot work if the images are not
                # mapprojected and thus not being shown.
                rescale_factor = 1
                left_image = np.zeros((1, 1))
                right_image = np.zeros((1, 1))

            self.plot_array(ax=axa[0], array=left_image, cmap="gray", add_cbar=False)
            axa[0].set_title(f"Left (n={match_point_df.shape[0]})")
            self.plot_array(ax=axa[1], array=right_image, cmap="gray", add_cbar=False)
            axa[1].set_title("Right (scenes shown only if mapprojected)")

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
        """
        Plot disparity maps from stereo processing.

        Creates a figure with three subplots showing the x and y components
        of the disparity map and the disparity magnitude, with optional
        quiver plot overlay.

        Parameters
        ----------
        unit : str, optional
            Unit for disparity values, either 'pixels' or 'meters', default is 'pixels'
        remove_bias : bool, optional
            Whether to remove the median offset from disparity values, default is True
        quiver : bool, optional
            Whether to overlay a quiver plot on the disparity magnitude plot,
            default is True
        save_dir : str or None, optional
            Directory to save the figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for the saved figure, default is None

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Raises
        ------
        ValueError
            If unit is not 'pixels' or 'meters'

        Notes
        -----
        The disparity map shows the pixel offset between corresponding points
        in the left and right images. This can be displayed in pixel units or
        converted to meters using the image GSD. The quiver plot shows the
        direction and magnitude of the disparity vectors.
        """
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

            # Combine masks from both bands to ensure consistency
            # This handles cases where one band might have valid data but the other doesn't
            combined_mask = np.ma.mask_or(dx.mask, dy.mask)
            dx.mask = combined_mask
            dy.mask = combined_mask

            full_gsd = Raster(self.disparity_fn).get_gsd()
            rescale_factor = sub_gsd / full_gsd
            dx = dx * rescale_factor
            dy = dy * rescale_factor

            if unit == "meters":
                dx = dx * full_gsd
                dy = dy * full_gsd

            if remove_bias:
                dx_offset = np.ma.median(dx)
                dy_offset = np.ma.median(dy)
                dx = dx - dx_offset
                dy = dy - dy_offset

            # Compute magnitude while preserving mask (combine masks from both dx and dy)
            dm = np.ma.sqrt(dx**2 + dy**2)
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
        """
        Create a detailed plot with DEM hillshade and subsets.

        Generates a detailed figure showing the full DEM hillshade with
        color overlay, plus three smaller subsets highlighting areas with
        different levels of intersection error. If the images are map-projected,
        it also shows the corresponding optical image for each subset.

        If the intersection error file is missing, it falls back to a plain
        hillshade plot without the detailed subsets.

        Parameters
        ----------
        intersection_error_percentiles : list, optional
            Percentiles of intersection error to use for selecting subsets,
            default is [16, 50, 84]
        subset_km : float, optional
            Size of the subset areas in kilometers, default is 1
        save_dir : str or None, optional
            Directory to save the figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for the saved figure, default is None

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        This method creates a detailed visualization with a large overview
        map at the top and six smaller subplots below (three hillshades and
        three optical images). The subsets are chosen based on the variance
        of intersection error, representing areas with different quality
        levels in the DEM.
        """
        if not self.intersection_error_fn:
            logger.warning(
                "\n\nIntersection error file not found. Plotting hillshade without details.\n\n"
            )
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=220)
            fig.suptitle(self.title, size=10)

            raster = Raster(self.dem_fn)
            dem = raster.read_array()
            gsd = raster.get_gsd()
            hs = raster.hillshade()
            self._plot_hillshade_with_overlay(ax, dem, hs, gsd)

            fig.tight_layout()
            if save_dir and fig_fn:
                save_figure(fig, save_dir, fig_fn)

            return

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
        # We only show the corresponding image if it is mapprojected
        if self.orthos:
            image = Raster(self.left_image_fn)

        # Full hillshade with DEM overlay
        self._plot_hillshade_with_overlay(ax_top, dem, hs, gsd)
        ax_top.set_title(self.title, size=14)

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
            self._plot_hillshade_with_overlay(ax_hs, dem_subset, hs_subset, gsd)

            with rio.open(self.dem_fn) as src:
                transform = src.transform
                ul_x, ul_y = rio.transform.xy(
                    transform, idx[0] * subset_size, idx[1] * subset_size
                )
                lr_x, lr_y = rio.transform.xy(
                    transform, (idx[0] + 1) * subset_size, (idx[1] + 1) * subset_size
                )

            # We only show the corresponding image if it is mapprojected
            if self.orthos:
                image_subset = image.read_raster_subset((ul_x, lr_y, lr_x, ul_y))
                # Use masked array operations to exclude nodata values from clim calculation
                clim = [
                    np.ma.min(image_subset),
                    np.percentile(image_subset.compressed(), 95),
                ]
                self.plot_array(
                    ax=ax_img,
                    array=image_subset,
                    clim=clim,
                    cmap="gray",
                    add_cbar=False,
                )
                axes_to_modify = [ax_hs, ax_img]
            else:
                plt.delaxes(ax_img)
                axes_to_modify = [ax_hs]

            for ax in axes_to_modify:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(None)
                for spine in ax.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(4)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def get_diff_vs_reference(self):
        """
        Get the difference between the DEM and a reference DEM.

        Tries to find an existing DEM difference file, or generates
        a difference on-the-fly if a reference DEM is available.

        Returns
        -------
        numpy.ma.MaskedArray or None
            Masked array containing the difference between the DEM and
            reference DEM, or None if no reference DEM is available

        Notes
        -----
        This method first looks for a DEM difference file with a pattern
        like *DEM*diff.tif. If found, it uses that. Otherwise, it computes
        the difference between the DEM and reference DEM on-the-fly using
        the Raster.compute_difference method.
        """
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
                diff = Raster(self.dem_fn).compute_difference(
                    self.reference_dem, save=True
                )
        return diff

    def _plot_hillshade_with_overlay(
        self, ax, dem, hillshade, gsd, clim=None, add_scalebar=True
    ):
        """
        Plot hillshade with semi-transparent DEM overlay.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        dem : numpy.ndarray
            DEM array to overlay
        hillshade : numpy.ndarray
            Hillshade array (grayscale background)
        gsd : float
            Ground sample distance in meters for scalebar
        clim : tuple or None, optional
            Color limits for DEM, default is None (auto)
        add_scalebar : bool, optional
            Whether to add a scalebar to the plot, default is True

        Notes
        -----
        This is a helper method to reduce code duplication across plotting
        methods. It creates the standard hillshade + DEM visualization used
        throughout the StereoPlotter class.
        """
        self.plot_array(ax=ax, array=hillshade, cmap="gray", add_cbar=False)
        self.plot_array(
            ax=ax,
            array=dem,
            clim=clim,
            cmap="viridis",
            cbar_label="Elevation (m HAE)",
            alpha=0.5,
        )
        if add_scalebar:
            scalebar = ScaleBar(gsd)
            ax.add_artist(scalebar)

    def plot_dem_results(
        self, el_clim=None, ie_clim=None, diff_clim=None, save_dir=None, fig_fn=None
    ):
        """
        Plot DEM results with hillshade, error, and difference maps.

        Creates a figure with three subplots showing the DEM with hillshade,
        intersection error, and difference with reference DEM.

        Parameters
        ----------
        el_clim : tuple or None, optional
            Color limits for elevation, default is None (auto)
        ie_clim : tuple or None, optional
            Color limits for intersection error, default is None (auto)
        diff_clim : tuple or None, optional
            Color limits for difference map, default is None (auto)
        save_dir : str or None, optional
            Directory to save the figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for the saved figure, default is None

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        This method creates a comprehensive visualization of the stereo
        DEM results, including:
        1. The DEM with hillshade overlay
        2. Triangulation intersection error
        3. Difference with reference DEM (if available)

        If any required files are missing, the corresponding subplot will
        display a message instead.
        """
        print("Plotting DEM results. This can take a minute for large inputs.")
        fig, axa = plt.subplots(1, 3, figsize=(10, 3), dpi=220)
        fig.suptitle(self.title, size=10)

        if self.dem_fn:
            raster = Raster(self.dem_fn)
            dem = raster.read_array()
            gsd = raster.get_gsd()
            hs = raster.hillshade()
            self._plot_hillshade_with_overlay(axa[0], dem, hs, gsd, clim=el_clim)
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
        axa[2].set_title("Reference DEM $-$ Stereo DEM")

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
