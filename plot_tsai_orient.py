#!/usr/bin/python

# Plot roll, pitch, and yaw of tsai cameras, before and after bundle adjustment.

# The naming convention used is that Forward images and cameras are
# named like ${prefix}f0000.tif, ${prefix}f0000.tsai, and analogously for Nadir and
# Aft (use 'n' and 'a'). If there is a dash before "f", it must be part of the prefix.

# To get cameras named this way, can create symlinks from original data.

# Inputs

# numCams=1000 # how many cameras to plot (if a large number is used, plot all)
# types="afn" # Plot only given types. Can be "afn", "af", "an", "a", etc. 
# beforeOpt="cam_json/" # cameras must be in the form ${beforeOpt}${x}.tsai, where x = a, f, n
# afterOpt="ba_triWt0.1_transWt1000_cam_json/run-" # Note the ending dash. Same convention as above.
# beforeCaption="PlanetOrig"
# afterCaption="BundleAdjust"
# subtractLineFit=1 # if to subtract the best line fit before plotting (same fit for before/after).

# Usage:

# python plot_tsai_orient.py $numCams $types $beforeOpt $afterOpt \
#    $beforeCaption $afterCaption $subtractLineFit

import sys, os, re
import matplotlib.pyplot as plt

import numpy as np
import os, sys, glob, shutil
from pyproj import Proj, transform, Transformer
from scipy.spatial.transform import Rotation as R

def produce_m(lon,lat,m_meridian_offset=0):
    """
    Produce M matrix which facilitates conversion from Lon-lat (NED) to ECEF coordinates
    From https://github.com/visionworkbench/visionworkbench/blob/master/src/vw/Cartography/Datum.cc#L249
    This is known as direction cosie matrix
    
    Parameters
    ------------
    lon: numeric
        longitude of spacecraft
    lat: numeric
        latitude of spacecraft
    m_meridian_offset: numeric
        set to zero
    Returns
    -----------
    R: np.array
        3 x 3 rotation matrix representing the m-matrix aka direction cosine matrix
    """
    if lat < -90:
        lat = -90
    if lat > 90:
        lat = 90
    
    rlon = (lon + m_meridian_offset) * (np.pi/180)
    rlat = lat * (np.pi/180)
    slat = np.sin(rlat)
    clat = np.cos(rlat)
    slon = np.sin(rlon)
    clon = np.cos(rlon)
    
    R = np.ones((3,3),dtype=float)
    R[0,0] = -slat*clon
    R[1,0] = -slat*slon
    R[2,0] = clat
    R[0,1] = -slon
    R[1,1] = clon
    R[2,1] = 0.0
    R[0,2] = -clon*clat
    R[1,2] = -slon*clat
    R[2,2] = -slat
    return R

def convert_ecef2NED(asp_rotation,lon,lat):
    """
    convert rotation matrices from ECEF to North-East-Down convention
    Parameters
    -------------
    asp_rotation: np.array
        3 x 3 rotation matrix from ASP
    lon: numeric
        longitude for computing m matrix
    lat: numeric
        latitude for computing m matrix
    
    Returns
    --------------
    r_ned: np.array
        3 x 3 NED rotation matrix 
    """
    m = produce_m(lon,lat)
    r_ned = np.matmul(np.linalg.inv(m),asp_rotation)  # this is the cam to ned transform
    #r_ned = np.matmul(np.transpose(m),asp_rotation)
    #r_ned = np.matmul(m,asp_rotation)
    return r_ned

def read_tsai_dict(tsai):
    """
    read tsai frame model from asp and return a python dictionary containing the parameters
    See ASP's frame camera implementation here: https://stereopipeline.readthedocs.io/en/latest/pinholemodels.html
    Parameters
    ----------
    tsai: str
        path to ASP frame camera model
    Returns
    ----------
    output: dictionary
        dictionary containing camera model parameters
    #TODO: support distortion model
    """
    camera = os.path.basename(tsai)
    with open(tsai, 'r') as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    fu = np.float64(content[2].split(' = ', 4)[1]) # focal length in x
    fv = np.float64(content[3].split(' = ', 4)[1]) # focal length in y
    cu = np.float64(content[4].split(' = ', 4)[1]) # optical center in x
    cv = np.float64(content[5].split(' = ', 4)[1]) # optical center in y
    cam = content[9].split(' = ', 10)[1].split(' ')
    cam_cen = [np.float64(x) for x in cam] # camera center coordinates in ECEF
    rot = content[10].split(' = ', 10)[1].split(' ')
    rot_mat = [np.float64(x) for x in rot] # rotation matrix for camera to world coordinates transformation
    pitch = np.float64(content[11].split(' = ', 10)[1]) # pixel pitch
    
    ecef_proj = 'EPSG:4978'
    geo_proj = 'EPSG:4326'
    ecef2wgs = Transformer.from_crs(ecef_proj, geo_proj)
    cam_cen_lat_lon = ecef2wgs.transform(cam_cen[0], cam_cen[1], cam_cen[2]) # this returns lat, lon and height
    # cam_cen_lat_lon = geolib.ecef2ll(cam_cen[0], cam_cen[1], cam_cen[2]) # camera center coordinates in geographic coordinates
    tsai_dict = {'camera':camera, 'focal_length':(fu, fv), 'optical_center':(cu, cv), 'cam_cen_ecef':cam_cen, 'cam_cen_wgs':cam_cen_lat_lon, 'rotation_matrix':rot_mat, 'pitch':pitch}
    return tsai_dict

def ned_rotation_from_tsai(tsai_fn):
    #coordinate conversion step
    from pyproj import Transformer
    ecef_proj = 'EPSG:4978'
    geo_proj = 'EPSG:4326'
    ecef2wgs = Transformer.from_crs(ecef_proj,geo_proj)
    
    # read tsai files
    asp_dict = read_tsai_dict(tsai_fn)
    
    # get camera position
    cam_cen = asp_dict['cam_cen_ecef']
    lat,lon,h = ecef2wgs.transform(*cam_cen)
    #print(lat,lon)
    # get camera rotation angle
    rot_mat = np.reshape(asp_dict['rotation_matrix'],(3,3))
    
   #rotate about z axis by 90 degrees
    #https://math.stackexchange.com/questions/651413/given-the-degrees-to-rotate-around-axis-how-do-you-come-up-with-rotation-matrix
    rot_z = np.zeros((3,3),float)
    angle = np.pi/2
    rot_z[0,0] = np.cos(angle) 
    rot_z[0,1] = -1 * np.sin(angle)
    rot_z[1,0] = np.sin(angle)
    rot_z[1,1] = np.cos(angle)
    rot_z[2,2] = 1
    
    #return np.matmul(rot_z,convert_ecef2NED(rot_mat,lon,lat))
    return R.from_matrix(np.matmul(rot_z,np.linalg.inv(convert_ecef2NED(rot_mat,lon,lat)))).as_euler('ZYX',degrees=True)

def poly_fit(X, Y):
    """
    Fit a polynomial of degree 1 and return the fitted Y values.
    """
    fit = np.poly1d(np.polyfit(X, Y, 1))
    return fit(X)

# Main function

if len(sys.argv) < 4:
    print("Usage: " + argv.sys[0] + " Num Types baDir")
    sys.exit(1)

Num   = int(sys.argv[1]) # How many to plot
Types = list(sys.argv[2]) # camera types, can be 'n', 'fna', etc.

# Assume cameras are named ${origPrefix}n1352.tsai and same for optPrefix
origPrefix = sys.argv[3]
optPrefix  = sys.argv[4]

origTag = sys.argv[5]
optTag = sys.argv[6]

subtractLineFit = int(sys.argv[7])

print("Camera types are: ", Types)
print("opt prefix ", optPrefix)
print("Subtract line fit: ", subtractLineFit)

f, ax = plt.subplots(3, 3, sharex=True, sharey = False, figsize = (8.5, 8))

print("types is ", Types)

count = -1
for s in Types:

    print("Loading: " + s)
    count += 1

    # Based on opt cameras find the original cameras. That because
    # maybe we optimized only a subset
    orig_cams = []
    opt_cams = sorted(glob.glob(optPrefix + s + '*.tsai'))
    if len(opt_cams) > Num:
        opt_cams = opt_cams[0:Num]
    for c in opt_cams:
        suff = c[-10:] # extract a1341.tsai
        orig_cams.append(origPrefix + suff)

    print("number of cameras for view " + s + ': ' + str(len(orig_cams)))
        
    if len(orig_cams) != len(opt_cams):
        print("Number of original and opt cameras must be the same")
        sys.exit(1) 

    currNum = Num
    currNum = min(len(orig_cams), currNum)
    orig_cams = orig_cams[0:currNum]
    opt_cams = opt_cams[0:currNum]
    
    # Get rotations, then convert to NED 
    orig_rotation_angles = np.array([ned_rotation_from_tsai(cam) for cam in orig_cams])
    opt_rotation_angles = np.array([ned_rotation_from_tsai(cam) for cam in opt_cams])

    orig_roll  = [r[2] for r in orig_rotation_angles]
    orig_pitch = [r[1] for r in orig_rotation_angles]
    orig_yaw   = [r[0] for r in orig_rotation_angles]

    opt_roll  = [r[2] for r in opt_rotation_angles]
    opt_pitch = [r[1] for r in opt_rotation_angles]
    opt_yaw   = [r[0] for r in opt_rotation_angles]

    residualTag = ''
    if subtractLineFit:
        fit_roll = poly_fit(np.array(range(len(orig_roll))), orig_roll)
        fit_pitch = poly_fit(np.array(range(len(orig_pitch))), orig_pitch)
        fit_yaw = poly_fit(np.array(range(len(orig_yaw))), orig_yaw)

        orig_roll = orig_roll - fit_roll
        orig_pitch = orig_pitch - fit_pitch
        orig_yaw = orig_yaw - fit_yaw
        
        opt_roll = opt_roll - fit_roll
        opt_pitch = opt_pitch - fit_pitch
        opt_yaw = opt_yaw - fit_yaw

        residualTag = ' residual'
        
    # Plot residuals after subtracting a linear fit
    ax[count][0].plot(np.arange(len(orig_roll)), orig_roll, label=origTag, color = 'r')
    ax[count][0].plot(np.arange(len(opt_roll)), opt_roll, label=optTag, color = 'b')

    ax[count][1].plot(np.arange(len(orig_pitch)), orig_pitch, label=origTag, color = 'r')
    ax[count][1].plot(np.arange(len(opt_pitch)), opt_pitch, label=optTag, color = 'b')

    ax[count][2].plot(np.arange(len(orig_yaw)), orig_yaw, label=origTag, color = 'r')
    ax[count][2].plot(np.arange(len(opt_yaw)), opt_yaw, label=optTag, color = 'b')

    if s == 'a':
        t = 'aft'
    if s == 'n':
        t = 'nadir'
    if s == 'f':
        t = 'fwd'

    ax[count][0].set_title(t + ' roll'  + residualTag)
    ax[count][1].set_title(t + ' pitch' + residualTag)
    ax[count][2].set_title(t + ' yaw '  + residualTag)

    ax[count][0].set_ylabel('Degrees')
    ax[count][1].set_ylabel('Degrees')
    ax[count][2].set_ylabel('Degrees')

    ax[count][0].set_xlabel('Frame number')
    ax[count][1].set_xlabel('Frame number')
    ax[count][2].set_xlabel('Frame number')

    ax[count][0].legend()
    ax[count][1].legend()
    ax[count][2].legend()

plt.tight_layout()
plt.show()


