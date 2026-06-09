import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from asp_plot.utils import Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Target longest-edge size (pixels) for "auto" downsampling of gallery thumbnails.
# Full-resolution ASP DEMs are often 15k+ pixels and hundreds of MB. This sets
# how much detail each thumbnail preserves; the save dpi is matched to it so one
# source pixel maps to ~one rendered pixel (crisp when zoomed) without bloating
# the file. 1200 px keeps a 1-to-N-panel gallery comfortably under ~10 MB.
GALLERY_TARGET_PX = 1200


class GalleryPlotter(Plotter):
    """
    Plot a grid of DEM thumbnails sharing a single color scale.

    This class extends the base Plotter class to visualize a *stack* of
    rasters (e.g. multi-date or multi-pair ASP DEMs over the same area) as a
    gallery of thumbnails. All panels share a single global percentile color
    stretch and one shared colorbar, which makes it easy to QA elevation
    differences across many outputs at a glance.

    DEMs are rendered using the same convention as the rest of the package
    (gray hillshade underlay with a semi-transparent ``viridis`` DEM on top).

    The layout sizes every panel to the rasters' actual aspect ratio and places
    panels with absolute positioning, so a gallery of 1 to N rasters (including
    non-square ones) packs tightly without stray whitespace. Per-panel titles
    use the full filename, auto-shrunk to fit the panel width.

    Attributes
    ----------
    raster_list : list of str
        Resolved list of raster file paths to plot.
    downsample : int or str
        Downsampling factor for reads, or "auto" to pick per-raster factors
        so the longest edge is ~``GALLERY_TARGET_PX`` pixels.
    title : str or None
        Figure suptitle, inherited from the Plotter class.

    Examples
    --------
    >>> gallery = GalleryPlotter.from_directory("/path/to/dems", pattern="*-DEM.tif")
    >>> gallery.plot_gallery(save_dir="/path/to/output", fig_fn="dem_gallery.png")
    """

    def __init__(self, raster_list, downsample="auto", **kwargs):
        """
        Initialize the GalleryPlotter object.

        Parameters
        ----------
        raster_list : list of str
            List of raster file paths to plot. Use the ``from_directory``
            classmethod to resolve a directory + glob pattern into this list.
        downsample : int or str, optional
            Downsampling factor passed to ``Raster``. "auto" (default) picks a
            per-raster factor so the longest edge is ~``GALLERY_TARGET_PX``.
        **kwargs : dict, optional
            Additional keyword arguments passed to the Plotter base class,
            particularly 'title' for the figure suptitle.

        Raises
        ------
        ValueError
            If ``raster_list`` is empty.
        """
        super().__init__(**kwargs)
        self.raster_list = [os.path.expanduser(fn) for fn in raster_list]
        if not self.raster_list:
            raise ValueError(
                "\n\nNo rasters to plot. Provide a non-empty list of files, "
                "or check the directory and pattern.\n\n"
            )
        self.downsample = downsample

    @classmethod
    def from_directory(cls, directory, pattern="*-DEM.tif", **kwargs):
        """
        Build a GalleryPlotter by globbing a directory for rasters.

        Parameters
        ----------
        directory : str
            Directory to search for rasters.
        pattern : str, optional
            Glob pattern for the rasters, default is "*-DEM.tif".
        **kwargs : dict, optional
            Additional keyword arguments passed to ``__init__`` (e.g.
            ``downsample``, ``title``).

        Returns
        -------
        GalleryPlotter
            Plotter initialized with the sorted list of matching files.

        Raises
        ------
        ValueError
            If no files in ``directory`` match ``pattern``.
        """
        directory = os.path.expanduser(directory)
        matches = glob_file(directory, pattern, all_files=True)
        if not matches:
            raise ValueError(
                f"\n\nNo files matching '{pattern}' found in {directory}.\n\n"
            )
        return cls(sorted(matches), **kwargs)

    @staticmethod
    def _grid_shape(n, aspect=1.0):
        """
        Compute the (nrows, ncols) grid for ``n`` panels of a given aspect.

        Searches all column counts and picks the one whose overall grid is
        closest to square in display space (accounting for each panel's
        width/height ``aspect``), lightly penalizing empty trailing cells.
        This keeps galleries of 1 to N panels — square or not — visually
        balanced.

        Parameters
        ----------
        n : int
            Number of panels.
        aspect : float, optional
            Panel width / height ratio, default 1.0 (square).

        Returns
        -------
        tuple of int
            (nrows, ncols)
        """
        best = None
        for ncols in range(1, n + 1):
            nrows = int(np.ceil(n / ncols))
            display_aspect = (ncols * aspect) / nrows
            empty = nrows * ncols - n
            # log() so e.g. 2:1 and 1:2 are penalized equally around square.
            cost = abs(np.log(display_aspect)) + 0.1 * empty
            if best is None or cost < best[0]:
                best = (cost, (nrows, ncols))
        return best[1]

    @staticmethod
    def _fit_title_fontsize(text, panel_w_in, max_fs=8.0, min_fs=4.0):
        """
        Pick a font size so ``text`` fits within ``panel_w_in`` inches.

        Uses an average character-width estimate (~0.58 * fontsize in points)
        rather than a renderer, so it works headlessly and deterministically.

        Parameters
        ----------
        text : str
            The title text.
        panel_w_in : float
            Panel width in inches.
        max_fs, min_fs : float, optional
            Font-size clamps in points.

        Returns
        -------
        float
            Font size in points.
        """
        panel_w_pts = panel_w_in * 72.0
        fs = panel_w_pts / (0.58 * max(len(text), 1))
        return float(np.clip(fs, min_fs, max_fs))

    def _resolve_downsample(self, ds):
        """
        Resolve the downsample factor for a single dataset.

        Returns ``self.downsample`` directly if it is an integer, otherwise
        ("auto") picks a factor so the longest edge is ~``GALLERY_TARGET_PX``.
        """
        if isinstance(self.downsample, str):
            longest = max(ds.height, ds.width)
            return max(1, int(np.ceil(longest / GALLERY_TARGET_PX)))
        return self.downsample

    def _hillshade_at(self, fn, shape):
        """
        Compute a GDAL hillshade for ``fn`` matching a downsampled DEM shape.

        Mirrors ``Raster.hillshade()`` (GDAL ``hillshade``, computeEdges) but
        runs on a downsampled in-memory copy so the underlay array matches the
        downsampled DEM array in shape.

        Parameters
        ----------
        fn : str
            Path to the DEM.
        shape : tuple of int
            Target (height, width) of the downsampled DEM.

        Returns
        -------
        numpy.ma.MaskedArray
            Hillshade array (nodata masked at 0).
        """
        out_h, out_w = shape
        gdal_ds = gdal.Open(fn)
        mem = gdal.Translate("", gdal_ds, format="MEM", width=out_w, height=out_h)
        hs_ds = gdal.DEMProcessing(
            "", mem, "hillshade", format="MEM", computeEdges=True
        )
        return np.ma.masked_equal(hs_ds.ReadAsArray(), 0)

    def plot_gallery(
        self,
        hillshade=True,
        cmap="viridis",
        clim=None,
        cbar_label="Elevation (m HAE)",
        panel_size=3.0,
        dpi=None,
        max_dpi=600,
        max_filesize_mb=10.0,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot the DEM gallery.

        Reads every raster (downsampled), computes a global shared color
        stretch, and lays the thumbnails out in an aspect-matched grid with one
        shared colorbar. Each panel is titled with its filename, auto-shrunk to
        fit the panel width.

        Parameters
        ----------
        hillshade : bool, optional
            If True (default), draw a gray hillshade underlay beneath each DEM
            and render the DEM semi-transparently on top, matching the
            convention used elsewhere in the package.
        cmap : str, optional
            Colormap for the DEMs, default is "viridis".
        clim : tuple or None, optional
            Color limits (min, max). If None (default), a global percentile
            stretch shared across all rasters is computed via ``ColorBar``.
        cbar_label : str, optional
            Label for the shared colorbar, default is "Elevation (m HAE)".
        panel_size : float, optional
            Longest-edge size of each panel in inches, default 3.0.
        dpi : int or None, optional
            Save resolution. If None (default), matched to the rendered detail
            (so one source pixel ~ one image pixel), then lowered if needed to
            respect ``max_filesize_mb`` and clamped to ``max_dpi``.
        max_dpi : int, optional
            Upper bound on the auto dpi to keep file size in check, default 600.
        max_filesize_mb : float, optional
            Soft cap on the output PNG size in MB, default 10. The auto dpi is
            reduced so the total rendered pixel count stays within an estimated
            budget; this keeps galleries of many rasters from exceeding the cap.
            Ignored when ``dpi`` is given explicitly.
        save_dir : str or None, optional
            Directory to save the figure, default is None (don't save).
        fig_fn : str or None, optional
            Filename for the saved figure, default is None.

        Returns
        -------
        matplotlib.figure.Figure
            The gallery figure.
        """
        # Read every raster once (downsampled), caching arrays so we don't
        # re-read the (potentially huge) source files for plotting.
        arrays = []
        for fn in self.raster_list:
            ds_probe = Raster(fn)
            downsample = self._resolve_downsample(ds_probe.ds)
            arrays.append(Raster(fn, downsample=downsample).read_array())

        if clim is None:
            clim = self.cb.find_common_clim(arrays)

        n = len(arrays)

        # Panel box sized to the rasters' (median) aspect ratio so imshow's
        # equal-aspect images fill each box with minimal internal whitespace.
        aspects = [a.shape[1] / a.shape[0] for a in arrays]
        aspect = float(np.median(aspects))
        if aspect >= 1.0:
            panel_w, panel_h = panel_size, panel_size / aspect
        else:
            panel_w, panel_h = panel_size * aspect, panel_size

        nrows, ncols = self._grid_shape(n, panel_w / panel_h)

        # Absolute (inches) layout so trailing empty cells leave no gap.
        title_h = (
            self._fit_title_fontsize(
                max(os.path.basename(fn) for fn in self.raster_list), panel_w
            )
            / 72.0
            + 0.06
        )
        left, right, bottom = 0.12, 0.12, 0.12
        top = 0.12 + (0.3 if self.title else 0.0)
        wgap, hgap = 0.14, title_h + 0.06
        cbar_gap, cbar_w = 0.18, 0.22

        grid_w = ncols * panel_w + (ncols - 1) * wgap
        grid_h = nrows * panel_h + (nrows - 1) * hgap
        fig_w = left + grid_w + cbar_gap + cbar_w + right
        fig_h = top + grid_h + bottom

        # Match save dpi to the actual rendered detail (median longest edge),
        # then lower it if needed so the total rendered pixel count stays within
        # a file-size budget (keeps many-raster galleries under max_filesize_mb).
        if dpi is None:
            med_long_px = float(np.median([max(a.shape) for a in arrays]))
            detail_dpi = round(med_long_px / panel_size)
            # ~0.8 bytes/pixel is a conservative estimate for these hillshaded
            # DEM PNGs; the 0.85 factor adds headroom against the soft cap.
            budget_px = 0.85 * max_filesize_mb * 1e6 / 0.8
            budget_dpi = (budget_px / (fig_w * fig_h)) ** 0.5
            dpi = int(np.clip(min(detail_dpi, budget_dpi), 100, max_dpi))

        fig = plt.figure(figsize=(fig_w, fig_h))
        if self.title is not None:
            fig.suptitle(self.title, size=10)

        im = None
        for i, (fn, array) in enumerate(zip(self.raster_list, arrays)):
            row, col = divmod(i, ncols)
            x_in = left + col * (panel_w + wgap)
            y_in = fig_h - top - row * (panel_h + hgap) - panel_h
            ax = fig.add_axes(
                [x_in / fig_w, y_in / fig_h, panel_w / fig_w, panel_h / fig_h]
            )
            if hillshade:
                hs = self._hillshade_at(fn, array.shape)
                self.plot_array(ax=ax, array=hs, cmap="gray", add_cbar=False)
                im = self.plot_array(
                    ax=ax,
                    array=array,
                    clim=clim,
                    cmap=cmap,
                    add_cbar=False,
                    alpha=0.5,
                )
            else:
                im = self.plot_array(
                    ax=ax, array=array, clim=clim, cmap=cmap, add_cbar=False
                )
            fontsize = self._fit_title_fontsize(os.path.basename(fn), panel_w)
            ax.set_title(os.path.basename(fn), size=fontsize)

        # Single shared colorbar spanning the panel grid on the right.
        if im is not None:
            cbar_x = left + grid_w + cbar_gap
            cax = fig.add_axes(
                [cbar_x / fig_w, bottom / fig_h, cbar_w / fig_w, grid_h / fig_h]
            )
            cbar = fig.colorbar(
                im,
                cax=cax,
                extend=self.cb.get_cbar_extend(arrays[0], clim),
            )
            cbar.set_label(cbar_label)

        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn, dpi=dpi)

        return fig
