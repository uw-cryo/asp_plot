#! /usr/bin/env python

import numpy as np
import rasterio as rio
from rasterio.windows import Window

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

def read_array(fn, b=1):
    ds = rio.open(fn)
    a = ds.read(b, masked=True)
    ndv = get_ndv(ds)
    #The stddev grids have nan values, separate from nodata
    ma = np.ma.fix_invalid(np.ma.masked_equal(a, ndv))
    return ma

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