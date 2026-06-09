import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from asp_plot.utils import Plotter, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Target longest-edge size (pixels) for "auto" downsampling of gallery thumbnails.
# Full-resolution ASP DEMs are often 15k+ pixels and hundreds of MB; thumbnails
# only need a few hundred pixels per panel.
GALLERY_TARGET_PX = 800


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
    def _grid_shape(n, width, height):
        """
        Compute the (nrows, ncols) grid for ``n`` panels given a figure aspect.

        Ported from the legacy gallery script: chooses a column count that
        keeps panels roughly square for the given figure width/height.

        Parameters
        ----------
        n : int
            Number of panels.
        width : float
            Figure width.
        height : float
            Figure height.

        Returns
        -------
        tuple of int
            (nrows, ncols)
        """
        ncols = int(np.ceil(np.sqrt((float(n) * width) / height)))
        nrows = int(np.ceil(float(n) / ncols))
        return nrows, ncols

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

    def _hillshade_at(self, fn, downsample, shape):
        """
        Compute a GDAL hillshade for ``fn`` matching a downsampled DEM shape.

        Mirrors ``Raster.hillshade()`` (GDAL ``hillshade``, computeEdges) but
        runs on a downsampled in-memory copy so the underlay array matches the
        downsampled DEM array in shape.

        Parameters
        ----------
        fn : str
            Path to the DEM.
        downsample : int
            Downsample factor (unused directly; ``shape`` drives the size).
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
        figsize=(7.5, 10.0),
        dpi=300,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot the DEM gallery.

        Reads every raster (downsampled), computes a global shared color
        stretch, and lays the thumbnails out in an auto-sized grid with one
        shared colorbar. Each panel is titled with its filename.

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
        figsize : tuple of float, optional
            Figure size in inches, default is (7.5, 10.0).
        dpi : int, optional
            Figure resolution, default is 300.
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
        downsamples = []
        for fn in self.raster_list:
            ds_probe = Raster(fn)
            downsample = self._resolve_downsample(ds_probe.ds)
            raster = Raster(fn, downsample=downsample)
            arrays.append(raster.read_array())
            downsamples.append(downsample)

        if clim is None:
            clim = self.cb.find_common_clim(arrays)

        n = len(arrays)
        width, height = figsize
        nrows, ncols = self._grid_shape(n, width, height)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
        axes = np.atleast_1d(axes).ravel()
        if self.title is not None:
            fig.suptitle(self.title, size=10)

        im = None
        for i, (fn, array, downsample) in enumerate(
            zip(self.raster_list, arrays, downsamples)
        ):
            ax = axes[i]
            if hillshade:
                hs = self._hillshade_at(fn, downsample, array.shape)
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
                    ax=ax,
                    array=array,
                    clim=clim,
                    cmap=cmap,
                    add_cbar=False,
                )
            ax.set_title(os.path.basename(fn), size=5)

        # Turn off any unused axes in the grid.
        for ax in axes[n:]:
            ax.axis("off")

        # Single shared colorbar across all panels.
        if im is not None:
            cbar = fig.colorbar(
                im,
                ax=axes.tolist(),
                extend=self.cb.get_cbar_extend(arrays[0], clim),
                fraction=0.025,
                pad=0.02,
            )
            cbar.set_label(cbar_label)

        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

        return fig
