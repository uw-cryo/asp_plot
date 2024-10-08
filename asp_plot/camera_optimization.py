#
# Methods below copied from orbit_plot.py in the ASP source code on 15-Aug-2024:
# https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/orbit_plot.py
#
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R


def read_frame_cam_dict(cam):

    # Invoke the appropriate reader for .tsai and .json frame cameras
    if cam.endswith(".tsai"):
        return read_tsai_cam(cam)
    elif cam.endswith(".json"):
        return read_frame_csm_cam(cam)
    else:
        raise Exception("Unknown camera file extension: " + cam)


def estim_satellite_orientation(positions):
    """
    Given a list of satellite positions, estimate the satellite
    orientation at each position. The x axis is the direction of
    motion, z points roughly down while perpendicular to x, and
    y is the cross product of z and x.
    """
    num = len(positions)
    rotations = []
    for i in range(num):
        prev_i = i - 1
        if prev_i < 0:
            prev_i = 0
        next_i = i + 1
        if next_i >= num:
            next_i = num - 1

        # x is tangent to orbit, z goes down
        x = np.array(positions[next_i]) - np.array(positions[prev_i])
        z = -np.array(positions[i])

        # Normalize
        z = z / np.linalg.norm(z)
        x = x / np.linalg.norm(x)

        # Make sure z is perpendicular to x
        z = z - np.dot(z, x) * x
        z = z / np.linalg.norm(z)

        # Find y as the cross product
        y = np.cross(z, x)

        # Make these as columns in a matrix r
        r = np.column_stack((x, y, z))
        rotations.append(r)

    return rotations


def read_tsai_cam(tsai):
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
    with open(tsai, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    fu = np.float64(content[2].split(" = ", 4)[1])  # focal length in x
    fv = np.float64(content[3].split(" = ", 4)[1])  # focal length in y
    cu = np.float64(content[4].split(" = ", 4)[1])  # optical center in x
    cv = np.float64(content[5].split(" = ", 4)[1])  # optical center in y
    cam = content[9].split(" = ", 10)[1].split(" ")
    cam_cen = [np.float64(x) for x in cam]  # camera center coordinates in ECEF
    rot = content[10].split(" = ", 10)[1].split(" ")
    rot_mat = [
        np.float64(x) for x in rot
    ]  # rotation matrix for camera to world coordinates transformation

    # Reshape as 3x3 matrix
    rot_mat = np.reshape(rot_mat, (3, 3))

    pitch = np.float64(content[11].split(" = ", 10)[1])  # pixel pitch
    tsai_dict = {
        "camera": camera,
        "focal_length": (fu, fv),
        "optical_center": (cu, cv),
        "cam_cen_ecef": cam_cen,
        "rotation_matrix": rot_mat,
        "pitch": pitch,
    }
    return tsai_dict


def read_frame_csm_cam(json_file):
    """
    Read rotation from a CSM Frame json state file.
    """

    with open(json_file, "r") as f:
        data = f.read()

    # Find first occurrence of open brace. This is needed because the CSM
    # state has some text before the JSON object.
    pos = data.find("{")
    # do substring from pos to the end, if pos was found
    if pos != -1:
        data = data[pos:]

    # parse the json from data
    j = json.loads(data)
    # print the json
    # print(json.dumps(j, indent=4, sort_keys=True))

    # Print all keys in the json
    # print("will print all keys in the json")
    # for key in j.keys():
    #     print(key)

    # Read the entry having the translation and rotation
    params = j["m_currentParameterValue"]

    # First three entries are the translation
    dict = {}
    dict["cam_cen_ecef"] = params[0:3]

    # Next four entries are the quaternion
    quat = params[3:7]

    # Convert the quaternion to rotation matrix
    r = R.from_quat(quat)
    mat = r.as_matrix()
    dict["rotation_matrix"] = mat

    return dict


def read_linescan_csm_cam(json_file):
    """
    Read rotation from a CSM linescan json state file.
    """

    with open(json_file, "r") as f:
        data = f.read()

    # Find first occurrence of open brace. This is needed because the CSM
    # state has some text before the JSON object.
    pos = data.find("{")
    # do substring from pos to the end, if pos was found
    if pos != -1:
        data = data[pos:]

    # parse the json from data
    j = json.loads(data)
    # print the json
    # print(json.dumps(j, indent=4, sort_keys=True))

    # Print all keys in the json
    # print("will print all keys in the json")
    # for key in j.keys():
    #     print(key)

    # Read the positions
    positions_vec = j["m_positions"]

    # Reshape to Nx3 matrix using the reshape function
    positions_vec = np.reshape(positions_vec, (-1, 3))

    # Create a vector of vectors
    positions = []
    for i in range(positions_vec.shape[0]):
        positions.append(positions_vec[i, :])

    # Read the quaternions
    quats = j["m_quaternions"]

    # Reshape to Nx4 matrix using the reshape function
    quats = np.reshape(quats, (-1, 4))

    # Iterate over the rows and convert to rotation matrix
    rotations = []
    for i in range(quats.shape[0]):
        r = R.from_quat(quats[i, :])
        # print the rotation matrix
        rotations.append(r.as_matrix())

    return (positions, rotations)


def isLinescan(cam_file):
    """
    Read the first line from cam_file which tells if the sensor is linescan.
    """
    lineScan = False
    with open(cam_file, "r") as f:
        line = f.readline()
        if "LINE_SCAN" in line:
            lineScan = True

    return lineScan


def roll_pitch_yaw(rot_mat, ref_rot_mat):

    # Rotate about z axis by 90 degrees. This must be synched up with
    # sat_sim. This will be a problem for non-sat_sim cameras.
    T = np.zeros((3, 3), float)
    T[0, 1] = -1
    T[1, 0] = 1
    T[2, 2] = 1
    Tinv = np.linalg.inv(T)

    inv_ref_rot_mat = np.linalg.inv(ref_rot_mat)
    N = np.matmul(inv_ref_rot_mat, rot_mat)

    return R.from_matrix(np.matmul(N, Tinv)).as_euler("XYZ", degrees=True)


def poly_fit(X, Y):
    """
    Fit a polynomial of degree 1 and return the fitted Y values.
    """
    fit = np.poly1d(np.polyfit(X, Y, 1))
    return fit(X)


# Read the positions and rotations from the given files. For linescan we will
# have a single camera, but with many poses in it. For Pinhole we we will have
# many cameras, each with a single pose.
def read_positions_rotations_from_file(cam_file):

    # Read the first line from cam_file
    lineScan = isLinescan(cam_file)

    positions = []
    rotations = []

    if lineScan:
        # Read linescan data
        (positions, rotations) = read_linescan_csm_cam(cam_file)
    else:
        # read Pinhole (Frame) files in ASP .tsai or CSM .json format
        asp_dict = read_frame_cam_dict(cam_file)
        # get camera rotation
        position = asp_dict["cam_cen_ecef"]
        rot_mat = asp_dict["rotation_matrix"]
        positions.append(position)
        rotations.append(rot_mat)

    return (positions, rotations)


# Read the positions and rotations from the given files
def read_positions_rotations(cams):

    (positions, rotations) = ([], [])
    for i in range(len(cams)):
        (p, r) = read_positions_rotations_from_file(cams[i])
        positions += p
        rotations += r

    # Must have as many rotations as positions. That is needed as later
    # we build ref rotations from positions.
    if len(rotations) != len(positions):
        print("Number of camera positions and orientations must be the same.")
        sys.exit(1)

    return (positions, rotations)


# Get rotations, then convert to NED.  That's why the loops below.
def read_angles(orig_cams, opt_cams, ref_cams):

    # orig_cams and ref_cams must be the same size
    if len(ref_cams) > 0 and len(orig_cams) != len(ref_cams):
        print(
            "Number of input and reference cameras must be the same. Got: ",
            len(ref_cams),
            " and ",
            len(opt_cams),
        )
        sys.exit(1)

    (orig_positions, orig_rotations) = read_positions_rotations(orig_cams)
    (opt_positions, opt_rotations) = read_positions_rotations(opt_cams)
    (ref_positions, ref_rotations) = read_positions_rotations(ref_cams)

    # If we do not have ref cameras that determine the satellite orientation,
    # estimate them from the camera positions
    if len(ref_cams) == 0:
        orig_ref_rotations = estim_satellite_orientation(orig_positions)
        opt_ref_rotations = estim_satellite_orientation(opt_positions)
    else:
        orig_ref_rotations = ref_rotations[:]
        opt_ref_rotations = ref_rotations[:]

    orig_rotation_angles = []
    opt_rotation_angles = []
    for i in range(len(orig_rotations)):
        angles = roll_pitch_yaw(orig_rotations[i], orig_ref_rotations[i])
        orig_rotation_angles.append(angles)
    for i in range(len(opt_rotations)):
        angles = roll_pitch_yaw(opt_rotations[i], opt_ref_rotations[i])
        opt_rotation_angles.append(angles)

    return (orig_rotation_angles, opt_rotation_angles)
