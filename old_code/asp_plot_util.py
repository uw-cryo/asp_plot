import numpy as np
import rasterio as rio
from rasterio import plot
from rasterio.windows import Window
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def symm_clim(clim):
    abs_max = np.max(np.abs(clim))
    return (-abs_max, abs_max)

def get_clim(ar, perc=(2,98), symm=False):
    try:
        clim = np.percentile(ar.compressed(), perc)
    except:
        clim = np.percentile(ar, perc)
    if symm:
        clim = symm_clim(clim)
    return clim

#Generalize for input list, not just two images
def find_common_clim(im1, im2, symm=False):
    clim1 = get_clim(im1)
    clim2 = get_clim(im2)
    clim = (np.min([clim1[0],clim2[0]]), np.max([clim1[1],clim2[1]]))
    if symm:
        clim = symm_clim(clim)
    return clim

def read_array(fn, b=1, extent=False):
    ds = rio.open(fn)
    a = ds.read(b, masked=True)
    ndv = get_ndv(ds)
    #The stddev grids have nan values, separate from nodata
    ma = np.ma.fix_invalid(np.ma.masked_equal(a, ndv))
    out = ma
    #Return map extent, should clean up to avoid different return
    if extent:
        extent = rio.plot.plotting_extent(ds)
        out = (ma, extent)
    return out

def get_ndv(ds):
    ndv = ds.nodatavals[0]
    if ndv == None:
        ndv = ds.read(1, window=Window(0, 0, 1, 1)).squeeze()
    return ndv

def get_cbar_extend(a, clim=None):
    """
    Determine whether we need to add triangles to ends of colorbar
    """
    if clim is None:
        clim = get_clim(a)
    extend = 'both'
    if a.min() >= clim[0] and a.max() <= clim[1]:
        extend = 'neither'
    elif a.min() >= clim[0] and a.max() > clim[1]:
        extend = 'max'
    elif a.min() < clim[0] and a.max() <= clim[1]:
        extend = 'min'
    return extend

def plot_ar(im, ax, cmap='inferno', clim=None, clim_perc=(2,98), label=None, cbar=True, alpha=1):
    if clim is None:
        clim = get_clim(im, clim_perc)
    m = ax.imshow(im, cmap=cmap, clim=clim, alpha=alpha, interpolation='none')
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad="2%")
        cb = plt.colorbar(m, cax=cax, ax=ax, extend=get_cbar_extend(im, clim))
        cax.set_ylabel(label)
    ax.set_facecolor("0.5")
    ax.set_xticks([])
    ax.set_yticks([])

