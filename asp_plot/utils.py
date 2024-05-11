import os
import numpy as np
import rasterio as rio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable


def show_existing_figure(filename):
    if os.path.exists(filename):
        img = mpimg.imread(filename)
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        print(f"Figure not found: {filename}")


class ColorBar:
    def __init__(self, perc_range=(2, 98), symm=False):
        self.perc_range = perc_range
        self.symm = symm

    def get_clim(self, input):
        try:
            clim = np.percentile(input.compressed(), self.perc_range)
        except:
            clim = np.percentile(input, self.perc_range)
        if self.symm:
            clim = self.symm_clim(clim)
        return clim

    def find_common_clim(self, inputs):
        clims = []
        for input in inputs:
            clim = self.get_clim(input)
            clims.append(clim)

        clim_min = np.min([clim[0] for clim in clims])
        clim_max = np.max([clim[1] for clim in clims])
        clim = (clim_min, clim_max)
        if self.symm:
            clim = self.symm_clim(clim)
        return clim

    def symm_clim(self, clim):
        abs_max = np.max(np.abs(clim))
        return (-abs_max, abs_max)

    def get_cbar_extend(self, input, clim=None):
        if clim is None:
            clim = self.get_clim(input)
        extend = "both"
        if input.min() >= clim[0] and input.max() <= clim[1]:
            extend = "neither"
        elif input.min() >= clim[0] and input.max() > clim[1]:
            extend = "max"
        elif input.min() < clim[0] and input.max() <= clim[1]:
            extend = "min"
        return extend


class Raster:
    def __init__(self, fn):
        self.fn = fn
        self.ds = rio.open(fn)

    def read_array(self, b=1, extent=False):
        a = self.ds.read(b, masked=True)
        ndv = self.get_ndv()
        ma = np.ma.fix_invalid(np.ma.masked_equal(a, ndv))
        out = ma
        if extent:
            extent = rio.plot.plotting_extent(self.ds)
            out = (ma, extent)
        return out

    def get_ndv(self):
        ndv = self.ds.nodatavals[0]
        if ndv == None:
            ndv = self.ds.read(1, window=Window(0, 0, 1, 1)).squeeze()
        return ndv


class Plotter:
    def __init__(
        self,
        cmap="inferno",
        clim=None,
        clim_perc=(2, 98),
        label=None,
        add_cbar=True,
        alpha=1,
    ):
        self.cmap = cmap
        self.clim = clim
        self.clim_perc = clim_perc
        self.label = label
        self.add_cbar = add_cbar
        self.alpha = alpha
        self.cb = ColorBar(perc_range=self.clim_perc)

    def plot_array(self, ax, array):
        if self.clim is None:
            self.clim = self.cb.get_clim(array)

        im = ax.imshow(
            array,
            cmap=self.cmap,
            clim=self.clim,
            alpha=self.alpha,
            interpolation="none",
        )

        if self.add_cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad="2%")
            plt.colorbar(
                im,
                cax=cax,
                ax=ax,
                extend=self.cb.get_cbar_extend(array, self.clim),
            )
            cax.set_ylabel(self.label)

        ax.set_facecolor("0.5")
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_geodataframe(self, ax, gdf, column_name, lognorm=False):
        if self.clim is None:
            self.clim = self.cb.get_clim(gdf[column_name])
        vmin, vmax = self.clim

        if lognorm:
            norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        gdf.plot(
            ax=ax,
            column=column_name,
            cmap=self.cmap,
            norm=norm,
            s=1,
            legend=True,
            legend_kwds={"label": self.label},
        )
