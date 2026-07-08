import logging
import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from matplotlib_scalebar.scalebar import ScaleBar

from asp_plot.processing_parameters import ProcessingParameters
from asp_plot.selections import (
    bbox_to_pixel_offset,
    pixel_window_to_bbox,
    reproject_bbox,
)
from asp_plot.utils import (
    ColorBar,
    Plotter,
    Raster,
    detect_satellite_attribution,
    glob_file,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class StereoFiles:
    """
    Discover the ASP stereo output files for a processing directory.

    This isolates file discovery (globbing for DEMs, disparity maps, match
    files, reference DEM, etc.) from the plotting in ``StereoPlotter``,
    mirroring the ``ReadBundleAdjustFiles`` / ``PlotBundleAdjustFiles`` split.
    The resolved paths and flags are exposed as plain attributes, which
    ``StereoPlotter`` consumes.

    Attributes
    ----------
    directory : str
        Root directory for ASP processing.
    stereo_directory : str
        Directory containing stereo processing outputs.
    full_directory : str
        Full path to the stereo directory.
    attribution : str or None
        Rights-holder of the source imagery ("Vantor", "Airbus DS", ...);
        gates the copyright overlay on scene panels.
    reference_dem : str or None
        Path to the reference DEM (supplied or recovered from the stereo log).
    left_image_fn, left_image_sub_fn, right_image_sub_fn : str or None
        Left/right (sub-sampled) image paths.
    orthos : bool
        Whether the left image is map-projected.
    align_left_fn, align_right_fn : str or None
        Alignment transform text files.
    match_point_fn : str
        Match-point file (the non-``-disp-`` one when several exist).
    disparity_sub_fn, disparity_fn : str or None
        Sub-sampled and full disparity map paths.
    dem_gsd : float or None
        Ground sample distance of the DEM in meters.
    dem_fn : str
        Path to the stereo DEM.
    intersection_error_fn : str or None
        Triangulation intersection-error raster.
    """

    def __init__(
        self,
        directory,
        stereo_directory,
        dem_gsd=None,
        dem_fn=None,
        reference_dem=None,
    ):
        """
        Discover stereo output files.

        Parameters
        ----------
        directory : str
            Root directory for ASP processing.
        stereo_directory : str
            Directory containing stereo processing outputs.
        dem_gsd : float, optional
            Ground sample distance of the DEM in meters.
        dem_fn : str, optional
            Path to the DEM file, default is None (automatically detected).
        reference_dem : str, optional
            Path to the reference DEM file, default is None (recovered from the
            stereo log).

        Raises
        ------
        ValueError
            If no DEM file is found in the stereo directory.
        """
        self.directory = os.path.expanduser(directory)
        self.stereo_directory = stereo_directory
        self.attribution = detect_satellite_attribution(self.directory)

        if reference_dem:
            self.reference_dem = os.path.expanduser(reference_dem)
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
            # If the reference DEM path from the log is relative, make it absolute
            elif not os.path.isabs(self.reference_dem):
                self.reference_dem = os.path.join(self.directory, self.reference_dem)
        if self.reference_dem:
            print(f"\nReference DEM: {self.reference_dem}\n")

        self.full_directory = os.path.join(self.directory, self.stereo_directory)
        self.left_image_fn = glob_file(self.full_directory, "*-L.tif")
        # Set processing flag if the left image is not mapprojected
        self.orthos = False if Raster(self.left_image_fn).transform is None else True
        self.left_image_sub_fn = glob_file(self.full_directory, "*-L_sub.tif")
        self.right_image_sub_fn = glob_file(self.full_directory, "*-R_sub.tif")
        self.align_left_fn = glob_file(self.full_directory, "*-align-L.txt")
        self.align_right_fn = glob_file(self.full_directory, "*-align-R.txt")

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
        stereo directory with a pattern like *-DEM.tif or *_dem.tif. File
        discovery is delegated to :class:`StereoFiles`; the resolved paths are
        exposed as read-only properties on this plotter.
        """
        self.files = StereoFiles(
            directory,
            stereo_directory,
            dem_gsd=dem_gsd,
            dem_fn=dem_fn,
            reference_dem=reference_dem,
        )
        super().__init__(attribution=self.files.attribution, **kwargs)

    # File discovery lives in StereoFiles; expose the resolved paths/flags as
    # read-only properties so the plotting methods can keep using ``self.<attr>``.
    @property
    def directory(self):
        return self.files.directory

    @property
    def stereo_directory(self):
        return self.files.stereo_directory

    @property
    def full_directory(self):
        return self.files.full_directory

    @property
    def reference_dem(self):
        return self.files.reference_dem

    @property
    def left_image_fn(self):
        return self.files.left_image_fn

    @property
    def orthos(self):
        return self.files.orthos

    @property
    def left_image_sub_fn(self):
        return self.files.left_image_sub_fn

    @property
    def right_image_sub_fn(self):
        return self.files.right_image_sub_fn

    @property
    def align_left_fn(self):
        return self.files.align_left_fn

    @property
    def align_right_fn(self):
        return self.files.align_right_fn

    @property
    def match_point_fn(self):
        return self.files.match_point_fn

    @property
    def disparity_sub_fn(self):
        return self.files.disparity_sub_fn

    @property
    def disparity_fn(self):
        return self.files.disparity_fn

    @property
    def dem_gsd(self):
        return self.files.dem_gsd

    @property
    def dem_fn(self):
        return self.files.dem_fn

    @property
    def intersection_error_fn(self):
        return self.files.intersection_error_fn

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
        subsampled images with match points overlaid as small red circles.
        For mapprojected scenes, match points are rescaled using the GSD ratio.
        For non-mapprojected scenes, match points are transformed from original
        to aligned coordinate space using the alignment matrices, then rescaled
        to the subsampled image dimensions.

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
        """
        match_point_df = self.get_match_point_df()

        fig, axa = plt.subplots(1, 2, figsize=(10, 5))

        if (
            self.left_image_sub_fn
            and self.right_image_sub_fn
            and match_point_df is not None
        ):
            if self.orthos:
                full_gsd = Raster(self.left_image_fn).get_gsd()
                sub_gsd = Raster(self.left_image_sub_fn).get_gsd()
                rescale_factor = sub_gsd / full_gsd
                left_x = match_point_df["x1"] / rescale_factor
                left_y = match_point_df["y1"] / rescale_factor
                right_x = match_point_df["x2"] / rescale_factor
                right_y = match_point_df["y2"] / rescale_factor
            else:
                if not self.align_left_fn or not self.align_right_fn:
                    raise FileNotFoundError(
                        "Alignment matrix files (run-align-{L,R}.txt) not found. "
                        "These are required to overlay match points on non-mapprojected images."
                    )

                full_width = Raster(self.left_image_fn).ds.width
                sub_width = Raster(self.left_image_sub_fn).ds.width
                rescale_factor = full_width / sub_width

                # Transform match points from original to aligned coordinate space
                align_L = np.loadtxt(self.align_left_fn)
                align_R = np.loadtxt(self.align_right_fn)

                n = len(match_point_df)
                ones = np.ones(n)

                left_pts = np.vstack([match_point_df["x1"], match_point_df["y1"], ones])
                left_aligned = align_L @ left_pts
                left_x = left_aligned[0] / rescale_factor
                left_y = left_aligned[1] / rescale_factor

                right_pts = np.vstack(
                    [match_point_df["x2"], match_point_df["y2"], ones]
                )
                right_aligned = align_R @ right_pts
                right_x = right_aligned[0] / rescale_factor
                right_y = right_aligned[1] / rescale_factor

            left_image = Raster(self.left_image_sub_fn).read_array()
            right_image = Raster(self.right_image_sub_fn).read_array()
            self.plot_array(
                ax=axa[0],
                array=left_image,
                cmap="gray",
                add_cbar=False,
                copyright=True,
            )
            self.plot_array(
                ax=axa[1],
                array=right_image,
                cmap="gray",
                add_cbar=False,
                copyright=True,
            )
            axa[0].set_title(f"Left (n={match_point_df.shape[0]})")
            axa[1].set_title("Right")

            axa[0].scatter(
                left_x,
                left_y,
                color="r",
                marker="o",
                facecolor="none",
                s=1,
            )
            axa[0].set_aspect("equal")

            axa[1].scatter(
                right_x,
                right_y,
                color="r",
                marker="o",
                facecolor="none",
                s=1,
            )
            axa[1].set_aspect("equal")
        else:
            self.plot_missing(axa[0])
            self.plot_missing(axa[1])

        fig.suptitle(self.title, size=10)
        self.save(fig, save_dir, fig_fn)

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
            dx = raster.read_array(b=1)
            dy = raster.read_array(b=2)

            # Combine masks from both bands to ensure consistency
            # This handles cases where one band might have valid data but the other doesn't
            combined_mask = np.ma.mask_or(dx.mask, dy.mask)
            dx.mask = combined_mask
            dy.mask = combined_mask

            # Rescale disparity from subsampled to full-res pixel coordinates.
            # Only meaningful for georeferenced (mapprojected) data where the
            # GSD ratio reflects the actual downsampling factor. For non-georeferenced
            # data the transform is identity/near-zero, so skip rescaling.
            if not self.orthos and unit == "meters":
                logger.warning(
                    "Disparity unit 'meters' not supported for non-mapprojected scenes; using pixels."
                )
            if self.orthos:
                sub_gsd = raster.get_gsd()
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

            if self.orthos:
                scalebar = ScaleBar(raster.get_gsd())
            else:
                scalebar = ScaleBar(1, units="px", dimension="pixel-length")
            axa[0].add_artist(scalebar)
            axa[0].set_title("x offset")
            axa[1].set_title("y offset")
            axa[2].set_title("offset magnitude")
        else:
            for ax in axa:
                self.plot_missing(ax)

        self.save(fig, save_dir, fig_fn)

    def _auto_hillshade_clip_offsets(
        self, ie, subset_size, intersection_error_percentiles
    ):
        """
        Select detailed-hillshade clip windows from intersection-error variance.

        Tiles the intersection-error raster into ``subset_size`` blocks,
        computes per-block variance (only for blocks that are >= 90% valid),
        and picks the blocks whose variance is closest to the requested
        percentiles (low/medium/high uncertainty).

        Parameters
        ----------
        ie : numpy.ma.MaskedArray
            Intersection-error array.
        subset_size : int
            Block size in pixels.
        intersection_error_percentiles : list
            Three percentiles for low/medium/high uncertainty blocks.

        Returns
        -------
        list of tuple
            Three ``(row_px, col_px)`` top-left pixel offsets.
        """
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

        offsets = []
        for pct in intersection_error_percentiles:
            target = np.percentile(compressed_variances, pct)
            idx = np.unravel_index(
                np.argmin(np.abs(block_variances - target)), block_variances.shape
            )
            offsets.append((idx[0] * subset_size, idx[1] * subset_size))
        return offsets

    def plot_detailed_hillshade(
        self,
        intersection_error_percentiles=[16, 50, 84],
        subset_km=1,
        clip_windows=None,
        clip_windows_crs=None,
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

        The clip boxes that were drawn are recorded on
        ``self.detailed_hillshade_clips`` (a list of
        ``{"label", "bbox", "pixel_offset"}`` dicts, ``bbox`` in DEM-CRS map
        coordinates) so a later run can replay the same clips via
        ``clip_windows`` for run-to-run comparison (issue #121).

        Parameters
        ----------
        intersection_error_percentiles : list, optional
            Percentiles of intersection error to use for selecting subsets,
            default is [16, 50, 84]
        subset_km : float, optional
            Size of the subset areas in kilometers, default is 1
        clip_windows : list or None, optional
            When provided, pins the subset clip boxes instead of selecting
            them from intersection-error variance. A list of up to three
            map-coordinate bounding boxes ``[xmin, ymin, xmax, ymax]`` (as
            written to a figure-selections file). Boxes that fall outside the
            current DEM fall back to the automatic selection with a warning.
            Default is None (automatic selection).
        clip_windows_crs : str or None, optional
            CRS the ``clip_windows`` bboxes are expressed in. When it differs
            from the current DEM's CRS, the boxes are reprojected first so the
            same ground area is clipped across stereo variants in different
            projections (e.g. mapprojected vs. non-mapprojected). Default is
            None (assume the boxes are already in the DEM's CRS).
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
        self.detailed_hillshade_clips = []

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

            self.save(fig, save_dir, fig_fn)

            return

        # Set up the plot
        fig = plt.figure(figsize=(12, 12), dpi=220, layout="constrained")
        gs = gridspec.GridSpec(3, 3, height_ratios=[2, 1, 1], wspace=0.1, figure=fig)

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
        rows, cols = ie.shape

        # Resolve the three clip windows as top-left pixel offsets. By default
        # they are selected from intersection-error variance (low/medium/high
        # uncertainty). When clip_windows is provided (e.g. replaying a prior
        # run's figure selections), pin those instead so reports are comparable.
        auto_offsets = None  # computed lazily; reused as fallback for bad boxes
        if clip_windows is None:
            auto_offsets = self._auto_hillshade_clip_offsets(
                ie, subset_size, intersection_error_percentiles
            )
            clip_offsets = list(auto_offsets)
        else:
            clip_offsets = []
            for i in range(3):
                if i < len(clip_windows) and clip_windows[i] is not None:
                    # Reproject the stored bbox to the current DEM's CRS if it
                    # was written in a different projection (e.g. mapprojected
                    # vs. non-mapprojected variant), so the same ground area is
                    # clipped. No-op when CRSs match or clip_windows_crs is None.
                    bbox = reproject_bbox(
                        clip_windows[i], clip_windows_crs, raster.ds.crs
                    )
                    row_px, col_px = bbox_to_pixel_offset(raster.ds.transform, bbox)
                    # Clamp small overflows; fall back to auto if a box lands
                    # entirely outside the (re-processed) DEM extent.
                    out_of_bounds = (
                        row_px >= rows or col_px >= cols or row_px < 0 or col_px < 0
                    )
                    if out_of_bounds:
                        logger.warning(
                            "\n\nPinned hillshade clip %d falls outside the DEM; "
                            "falling back to automatic selection.\n\n" % i
                        )
                        if auto_offsets is None:
                            auto_offsets = self._auto_hillshade_clip_offsets(
                                ie, subset_size, intersection_error_percentiles
                            )
                        clip_offsets.append(auto_offsets[i])
                    else:
                        row_px = min(row_px, max(rows - subset_size, 0))
                        col_px = min(col_px, max(cols - subset_size, 0))
                        clip_offsets.append((row_px, col_px))
                else:
                    if auto_offsets is None:
                        auto_offsets = self._auto_hillshade_clip_offsets(
                            ie, subset_size, intersection_error_percentiles
                        )
                    clip_offsets.append(auto_offsets[i])

        # Define distinct colors and labels for the rectangles and subplot axes
        rect_colors = ["magenta", "cyan", "orange"]
        clip_labels = ["low", "medium", "high"]

        # Record the clips (DEM-CRS bbox + pixel offset) so a later run can
        # replay them for run-to-run comparison (issue #121).
        self.detailed_hillshade_clips = [
            {
                "label": label,
                "bbox": pixel_window_to_bbox(
                    raster.ds.transform, row_px, col_px, subset_size, subset_size
                ),
                "pixel_offset": [int(row_px), int(col_px)],
            }
            for (row_px, col_px), label in zip(clip_offsets, clip_labels)
        ]

        # Add colored boxes outlining the three areas
        for (row_px, col_px), color in zip(clip_offsets, rect_colors):
            rect = plt.Rectangle(
                (col_px, row_px),
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
        for ax_hs, ax_img, (row_px, col_px), color in zip(
            axes_hillshade, axes_image, clip_offsets, rect_colors
        ):
            hs_subset = hs[
                row_px : row_px + subset_size,
                col_px : col_px + subset_size,
            ]
            dem_subset = dem[
                row_px : row_px + subset_size,
                col_px : col_px + subset_size,
            ]
            self._plot_hillshade_with_overlay(ax_hs, dem_subset, hs_subset, gsd)

            ul_x, ul_y = rio.transform.xy(raster.ds.transform, row_px, col_px)
            lr_x, lr_y = rio.transform.xy(
                raster.ds.transform,
                row_px + subset_size,
                col_px + subset_size,
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
                    copyright=True,
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

        self.save(fig, save_dir, fig_fn, tight_layout=False)

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
        like *diff.tif. If found, it uses that. Otherwise, it computes
        the difference between the DEM and reference DEM on-the-fly using
        the Raster.compute_difference method.
        """
        diff_fn = glob_file(self.full_directory, "*diff.tif")
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
            self.plot_missing(axa[0])
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
            self.plot_missing(axa[1])
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
            self.plot_missing(axa[2])
        axa[2].set_title("Reference DEM $-$ Stereo DEM")

        self.save(fig, save_dir, fig_fn)
