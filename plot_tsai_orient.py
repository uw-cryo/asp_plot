#!/usr/bin/python

# Plot roll, pitch, and yaw of tsai cameras, before and after bundle adjustment.

# The naming convention used is that Forward images and cameras are
# named like ${prefix}f0000.tif, ${prefix}f0000.tsai, and analogously for Nadir and
# Aft (use 'n' and 'a'). If there is a dash before "f", it must be part of the prefix.

# To get cameras named this way, can create symlinks from original data.

# For every camera named a-10000.tsai, there must be a reference camera named
# a-ref-10000.tsai. This is the camera that will be used to convert from ECEF
# to NED coordinates. This camera is created by sat_sim with the option
# --save-ref-cams. After bundle adjustment, such ref cams must be created manually,
# by copying the original ref cams and renaming them to fit the naming 
# convention for the bundle adjusted cameras.

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

import sys, os, re, math
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

def Rroll(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])
  
def Rpitch(theta):
  return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])
  
def Ryaw(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def ned_rotation_from_tsai(tsai_fn, ref_tsai_fn):
    #coordinate conversion step
    from pyproj import Transformer
    ecef_proj = 'EPSG:4978'
    geo_proj = 'EPSG:4326'
    ecef2wgs = Transformer.from_crs(ecef_proj,geo_proj)
    
    # read tsai files
    asp_dict = read_tsai_dict(tsai_fn)
    ref_asp_dict = read_tsai_dict(ref_tsai_fn)
    
    # get camera position
    cam_cen = asp_dict['cam_cen_ecef']
    lat,lon,h = ecef2wgs.transform(*cam_cen)
    # get camera rotation angle
    rot_mat = np.reshape(asp_dict['rotation_matrix'],(3,3))
    ref_rot_mat = np.reshape(ref_asp_dict['rotation_matrix'],(3,3))
    inv_ref_rot_mat = np.linalg.inv(ref_rot_mat)

    #rotate about z axis by 90 degrees
    #https://math.stackexchange.com/questions/651413/given-the-degrees-to-rotate-around-axis-how-do-you-come-up-with-rotation-matrix
    rot_z = np.zeros((3,3),float)
    angle = np.pi/2
    rot_z[0,0] = np.cos(angle) 
    rot_z[0,1] = -1 * np.sin(angle)
    rot_z[1,0] = np.sin(angle)
    rot_z[1,1] = np.cos(angle)
    rot_z[2,2] = 1
    
    T = np.zeros((3,3),float)
    T[0, 1] = 1
    T[1, 0] = -1
    T[2, 2] = 1
    Tinv = np.linalg.inv(T)

    N = np.matmul(inv_ref_rot_mat, rot_mat)

    # Verification that if the input rotations are done in order roll, pitch,
    # yaw, then the output of as_euler('XYZ') is the same as the input.
    #roll = 0.1; pitch = 0.2; yaw = 0.3 # in degrees
    #s = np.pi / 180.0 # convert to radians
    #Rt = Ryaw(yaw * s) * Rpitch(pitch * s) * Rroll(roll * s)
    #angles2 = R.from_matrix(Rt).as_euler('XYZ',degrees=True)
    #print("input is, ", [roll, pitch, yaw])

    angles = R.from_matrix(np.matmul(N, Tinv)).as_euler('XYZ',degrees=True)
    return angles

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
print("orig prefix ", origPrefix)
print("opt prefix ", optPrefix)
print("Subtract line fit: ", subtractLineFit)

f, ax = plt.subplots(3, 3, sharex=True, sharey = False, figsize = (15, 15))

# Set up the font for all elements
fs = 14
plt.rcParams.update({'font.size': fs})
plt.rc('axes', titlesize = fs)   # fontsize of the axes title
plt.rc('axes', labelsize = fs)   # fontsize of the x and y labels
plt.rc('xtick', labelsize = fs)  # fontsize of the tick labels
plt.rc('ytick', labelsize = fs)  # fontsize of the tick labels
plt.rc('legend', fontsize = fs)  # legend fontsize
plt.rc('figure', titlesize = fs) # fontsize of the figure title

count = -1
for s in Types:

    count += 1

    # Based on opt cameras find the original cameras. That because
    # maybe we optimized only a subset
    orig_cams = []
    all_opt_cams = sorted(glob.glob(optPrefix + s + '*.tsai'))
    ref_cams = sorted(glob.glob(optPrefix + s + '-ref-*.tsai'))
    camMap = set()
    # Add ref_cams to camMap set
    for c in ref_cams:
        camMap.add(c)
    opt_cams = []
    # Add to opt_cams only those cameras that are not in camMap
    # This is to avoid adding ref_cams to opt_cams
    for c in all_opt_cams:
        if c not in camMap:
            opt_cams.append(c) 
    # TODO(oalexan1): what if opt cams do not exist?

    # Check that ref cams and opt cams have same size
    if len(ref_cams) != len(opt_cams):
        print("Number of ref and opt cameras must be the same.")
        sys.exit(1)

    # Eliminate the ref cams from within opt cams
    if len(opt_cams) > Num:
        opt_cams = opt_cams[0:Num]
    for c in opt_cams:
        suff = c[len(optPrefix):] # extract a1341.tsai
        orig_cam = origPrefix + suff
        orig_cams.append(orig_cam)

    print("number of cameras for view " + s + ': ' + str(len(orig_cams)))
        
    if len(orig_cams) != len(opt_cams):
        print("Number of original and opt cameras must be the same")
        sys.exit(1)

    currNum = Num
    currNum = min(len(orig_cams), currNum)
    orig_cams = orig_cams[0:currNum]
    opt_cams = opt_cams[0:currNum]
    
    # Get rotations, then convert to NED 
    I = range(len(orig_cams))
    orig_rotation_angles = np.array([ned_rotation_from_tsai(orig_cams[i], ref_cams[i])
                                      for i in I])
    opt_rotation_angles = np.array([ned_rotation_from_tsai(opt_cams[i], ref_cams[i]) 
                                    for i in I])

    # The order is roll, pitch, yaw, as returned by R.from_matrix().as_euler('XYZ',degrees=True)
    orig_roll  = [r[0] for r in orig_rotation_angles]
    orig_pitch = [r[1] for r in orig_rotation_angles]
    orig_yaw   = [r[2] for r in orig_rotation_angles]
    opt_roll  = [r[0] for r in opt_rotation_angles]
    opt_pitch = [r[1] for r in opt_rotation_angles]
    opt_yaw   = [r[2] for r in opt_rotation_angles]

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

    if s == 'a':
        t = 'aft'
    if s == 'n':
        t = 'nadir'
    if s == 'f':
        t = 'fwd'

    print("stddev for " + origTag + " " + t + " roll: " + str(np.std(orig_roll)) + " degrees")
    print("stddev for " + origTag + " " + t + " pitch: " + str(np.std(orig_pitch)) + " degrees")
    print("stddev for " + origTag + " " + t + " yaw: " + str(np.std(orig_yaw)) + " degrees")

    print("stddev for " + optTag + " " + t + " roll: " + str(np.std(opt_roll)) + " degrees")
    print("stddev for " + optTag + " " + t + " pitch: " + str(np.std(opt_pitch)) + " degrees")
    print("stddev for " + optTag + " " + t + " yaw: " + str(np.std(opt_yaw)) + " degrees")

    # Plot residuals
    ax[count][0].plot(np.arange(len(orig_roll)), orig_roll, label=origTag, color = 'r')
    ax[count][0].plot(np.arange(len(opt_roll)), opt_roll, label=optTag, color = 'b')

    ax[count][1].plot(np.arange(len(orig_pitch)), orig_pitch, label=origTag, color = 'r')
    ax[count][1].plot(np.arange(len(opt_pitch)), opt_pitch, label=optTag, color = 'b')

    ax[count][2].plot(np.arange(len(orig_yaw)), orig_yaw, label=origTag, color = 'r')
    ax[count][2].plot(np.arange(len(opt_yaw)), opt_yaw, label=optTag, color = 'b')

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

    for index in range(3):
        ac = ax[count][index]
        for item in ([ac.title, ac.xaxis.label, ac.yaxis.label] +
             ac.get_xticklabels() + ac.get_yticklabels()):
          item.set_fontsize(fs)

plt.tight_layout()
plt.show()


