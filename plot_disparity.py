#! /usr/bin/env python

# from SB: https://github.com/uw-cryo/skysat_stereo/blob/master/scripts/plot_disparity.py

import os
import sys
import matplotlib.pyplot as plt
from pygeotools.lib import iolib
import glob
import numpy as np
from imview.lib import pltlib
def find_clim(im1,im2):
    perc1 = np.percentile(im1,(2,98))
    perc2 = np.percentile(im2,(2,98))
    perc = (np.min([perc1[0],perc2[0]]),np.max([perc1[1],perc2[1]]))
    abs_max = np.max(np.abs(perc))
    perc = (-abs_max,abs_max)
    return perc

dir_list = [os.path.abspath(x) for x in sys.argv[1:]]
for dir in dir_list:
    disparity_file = glob.glob(os.path.join(dir,'*-F.tif'))[0] #this is a multichannel file
    left_image_warped = glob.glob(os.path.join(dir,'*-L.tif'))[0] #1 channel
    right_image_warped = glob.glob(os.path.join(dir,'*-R.tif'))[0] #1 channel
    disp_ds = iolib.fn_getds(disparity_file)
    error_fn = glob.glob(os.path.join(dir,'*In*.tif'))[0]
    dem_fn = glob.glob(os.path.join(dir,'*-DEM.tif'))[0]
    dx = iolib.fn_getma(disparity_file,bnum=1)
    dy = iolib.fn_getma(disparity_file,bnum=2)
    img1 = iolib.fn_getma(left_image_warped)
    img2 = iolib.fn_getma(right_image_warped)
    error = iolib.fn_getma(error_fn)
    dem = iolib.fn_getma(dem_fn)
    dem_ds = iolib.fn_getds(dem_fn)
    base_dir = os.path.basename(dir)

    #title_str = dt_string+sat_string+img_string
    fig,ax = plt.subplots(3,2,figsize=(9,6))
    #print(disparity_file)
    #add code to create a fig
    #do not plot it, just save it as a png.
    #at some point, try to add a figure showing changes of disparity with elevation, might be good to add DEM as an input.
    #but that can be done only for map images
    pltlib.iv(img1,cmap='gray',title='Left',ax=ax[0,0],clim=(0,1))
    pltlib.iv(img2,cmap='gray',title='Right',ax=ax[0,1],clim=(0,1))
    clim = find_clim(dx,dy)
    pltlib.iv(dx,cmap='RdBu',title='dx',clim=clim,ax=ax[1,0])
    pltlib.iv(dy,cmap='RdBu',title='dy',clim=clim,ax=ax[1,1])
    pltlib.iv(error,cmap='plasma',title='Intersection error (m)',ax=ax[2,0])
    pltlib.iv(dem,ds=dem_ds,scalebar=True,title="Digital Elevation Model",hillshade=True,ax=ax[2,1])
    #fig.suptitle(title_str)
    fig.tight_layout()
    #outfn = dir+'convergence_{}_intersection_area_{}km2'.format(convergence_angle,intersection_area)+'.jpg'
    #print('Saving disparity plot at {} \n'.format(outfn))
    plt.show()

#fig.savefig(outfn,dpi=300)
print('Script Complete!')

