"""
Camera-model I/O helpers mirrored from the NASA Ames Stereo Pipeline (ASP).

The functions in this module are kept **function-based and as close to verbatim
as practical** to the upstream ``orbit_plot.py`` so they can be re-synced when
ASP changes. Do not refactor these into classes or rename them without a
corresponding upstream change -- that would make future syncing painful.

Upstream source:
https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/orbit_plot.py

Related CSM/CsmUtils references are noted inline on the individual functions.
"""

import json
import os
import sys

import geopandas as gpd
import numpy as np
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point

# Add this value to an ASP pixel to get a CSM pixel
ASP_TO_CSM_SHIFT = 0.5


def toCsmPixel(asp_pix):
    """
    Convert an ASP pixel coordinate to a CSM pixel coordinate.

    Parameters
    ----------
    asp_pix : array-like
        ASP pixel coordinates as [x, y]

    Returns
    -------
    numpy.ndarray
        CSM pixel coordinates

    Notes
    -----
    ASP and CSM use slightly different pixel coordinate conventions.
    CSM pixel coordinates are shifted by 0.5 pixels relative to ASP.
    Code copied from CsmModel.cc in ASP.
    """
    # Explicitly ensure csm_pix has float values even if the input may be int
    csm_pix = np.array([float(asp_pix[0]), float(asp_pix[1])])

    # Add the shift
    csm_pix[0] += ASP_TO_CSM_SHIFT
    csm_pix[1] += ASP_TO_CSM_SHIFT

    return csm_pix


def getTimeAtLine(model, line):
    """
    Find the time at a given line in a linescan camera model.

    Parameters
    ----------
    model : dict
        CSM camera model parameters
    line : int
        Line number in the image (0-based)

    Returns
    -------
    float
        Time corresponding to the given line

    Notes
    -----
    The time is computed using the linear mapping between line number
    and time defined in the CSM model. Code adapted from get_time_at_line()
    in CsmUtils.cc and getImageTime() in UsgsAstroLsSensorModel.cpp.
    """
    # Covert the line to a CSM pixel
    asp_pix = np.array([0.0, float(line)])
    csm_pix = toCsmPixel(asp_pix)

    referenceIndex = 0
    time = model["m_intTimeStartTimes"][referenceIndex] + model["m_intTimes"][
        referenceIndex
    ] * (csm_pix[1] - model["m_intTimeLines"][referenceIndex] + 0.5)

    return time


def getLineAtTime(time, model):
    """
    Get the line number at a given time in a linescan camera model.

    Parameters
    ----------
    time : float
        Time to convert to line number
    model : dict
        CSM camera model parameters

    Returns
    -------
    float
        Line number corresponding to the given time

    Raises
    ------
    Exception
        If the model does not have a linear relationship between time and lines

    Notes
    -----
    This function computes the line number in the image corresponding
    to the given time, assuming a linear relationship between time and
    line number. Code adapted from get_line_at_time() in CsmUtils.cc.
    """
    # All dt values in model['intTimes'] (slopes) must be equal, or else
    # the model is not linear in time.
    for i in range(1, len(model["m_intTimeLines"])):
        if abs(model["m_intTimes"][i] - model["m_intTimes"][0]) > 1e-10:
            raise Exception(
                "Expecting a linear relation between time and image lines.\n"
            )

    line0 = 0.0
    line1 = float(model["m_nLines"]) - 1.0
    time0 = getTimeAtLine(model, line0)
    time1 = getTimeAtLine(model, line1)

    return line0 + (line1 - line0) * (time - time0) / (time1 - time0)


def read_frame_cam_dict(cam):
    """
    Read a frame camera model file into a dictionary.

    Parameters
    ----------
    cam : str
        Path to the camera file (.tsai or .json)

    Returns
    -------
    dict
        Dictionary containing camera parameters

    Raises
    ------
    Exception
        If the file extension is not recognized
    """
    # Invoke the appropriate reader for .tsai and .json frame cameras
    if cam.endswith(".tsai"):
        return read_tsai_cam(cam)
    elif cam.endswith(".json"):
        return read_frame_csm_cam(cam)
    else:
        raise Exception("Unknown camera file extension: " + cam)


def estim_satellite_orientation(positions):
    """
    Estimate satellite orientation at each position.

    Parameters
    ----------
    positions : list of array-like
        List of satellite positions in ECEF coordinates

    Returns
    -------
    list of numpy.ndarray
        List of rotation matrices representing satellite orientation

    Notes
    -----
    For each position, computes a local coordinate system where:
    - x axis is the direction of motion
    - z points roughly down (towards Earth center)
    - y is perpendicular to both x and z
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


def read_csm_cam(json_file):
    """
    Read a CSM model state file in JSON format.

    Parameters
    ----------
    json_file : str
        Path to the CSM JSON state file

    Returns
    -------
    dict
        Dictionary containing the CSM model parameters

    Notes
    -----
    CSM JSON files sometimes have text before the actual JSON content.
    This function handles that by finding the first open brace and
    parsing the JSON from that point.
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

    return j


def read_tsai_cam(tsai):
    """
    Read a TSAI frame camera model into a dictionary.

    Parameters
    ----------
    tsai : str
        Path to ASP frame camera model file (.tsai)

    Returns
    -------
    dict
        Dictionary containing camera model parameters

    Notes
    -----
    TSAI is a camera model format used by ASP. It contains parameters such as
    focal length, optical center, camera position, and orientation. See ASP
    documentation for more details:
    https://stereopipeline.readthedocs.io/en/latest/pinholemodels.html
    """
    camera = os.path.basename(tsai)
    with open(tsai, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    fu = float(content[2].split(" = ", 4)[1])  # focal length in x
    fv = float(content[3].split(" = ", 4)[1])  # focal length in y
    cu = float(content[4].split(" = ", 4)[1])  # optical center in x
    cv = float(content[5].split(" = ", 4)[1])  # optical center in y
    cam = content[9].split(" = ", 10)[1].split(" ")
    cam_cen = [float(x) for x in cam]  # camera center coordinates in ECEF
    rot = content[10].split(" = ", 10)[1].split(" ")
    rot_mat = [
        float(x) for x in rot
    ]  # rotation matrix for camera to world coordinates transformation

    # Reshape as 3x3 matrix
    rot_mat = np.reshape(rot_mat, (3, 3))

    pitch = float(content[11].split(" = ", 10)[1])  # pixel pitch
    tsai_dict = {
        "camera": camera,
        "focal_length": (fu, fv),
        "optical_center": (cu, cv),
        "cam_cen_ecef": Point(cam_cen),
        "rotation_matrix": rot_mat,
        "pitch": pitch,
    }
    return tsai_dict


def tsai_list_to_gdf(tsai_fn_list):
    """
    Convert a list of TSAI camera files to a GeoDataFrame.

    Parameters
    ----------
    tsai_fn_list : list of str
        List of paths to TSAI camera files

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing camera model parameters with
        a Point geometry column for camera centers
    """
    tsai_dict_list = []
    for tsai_fn in tsai_fn_list:
        tsai_dict = read_tsai_cam(tsai_fn)
        tsai_dict_list.append(tsai_dict)

    gdf = gpd.GeoDataFrame(tsai_dict_list, geometry="cam_cen_ecef", crs="EPSG:4978")
    gdf.set_index("camera")

    return gdf


def read_frame_csm_cam(json_file):
    """
    Read position and orientation from a CSM Frame camera file.

    Parameters
    ----------
    json_file : str
        Path to the CSM JSON state file

    Returns
    -------
    dict
        Dictionary containing camera center and rotation matrix

    Notes
    -----
    This function extracts the camera position and orientation from a
    CSM frame camera model. The camera position is given by the first three
    parameters, and the orientation is represented as a quaternion that
    is converted to a rotation matrix.
    """
    j = read_csm_cam(json_file)

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


def read_linescan_pos_rot(json_file):
    """
    Read positions and rotations from a CSM linescan camera file.

    Parameters
    ----------
    json_file : str
        Path to the CSM JSON state file

    Returns
    -------
    tuple
        Tuple containing (positions, rotations) where:
        - positions is a list of camera positions
        - rotations is a list of rotation matrices

    Notes
    -----
    Linescan cameras have different positions and orientations for each
    line in the image. This function extracts the full list of positions
    and orientations from the CSM model.
    """
    j = read_csm_cam(json_file)

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
        rotations.append(r.as_matrix())

    return (positions, rotations)


def isLinescan(cam_file):
    """
    Check if a camera file is for a linescan sensor.

    Parameters
    ----------
    cam_file : str
        Path to the camera file

    Returns
    -------
    bool
        True if the camera is a linescan sensor, False otherwise

    Notes
    -----
    This function reads the first line of the camera file to check if
    it contains the string "LINE_SCAN", which indicates a linescan camera.
    """
    lineScan = False
    with open(cam_file, "r") as f:
        line = f.readline()
        if "LINE_SCAN" in line:
            lineScan = True

    return lineScan


def roll_pitch_yaw(rot_mat, ref_rot_mat):
    """
    Calculate roll, pitch, and yaw angles relative to a reference orientation.

    Parameters
    ----------
    rot_mat : numpy.ndarray
        Rotation matrix for the camera
    ref_rot_mat : numpy.ndarray
        Reference rotation matrix

    Returns
    -------
    numpy.ndarray
        Array of Euler angles [roll, pitch, yaw] in degrees

    Notes
    -----
    This function calculates the orientation of a camera relative to a
    reference orientation, and returns the result as Euler angles in
    roll, pitch, yaw (rotation around x, y, z axes) in degrees.
    """
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


def read_positions_rotations_from_file(cam_file):
    """
    Read positions and rotations from a camera file.

    Parameters
    ----------
    cam_file : str
        Path to the camera file

    Returns
    -------
    tuple
        Tuple containing (positions, rotations) where:
        - positions is a list of camera positions
        - rotations is a list of rotation matrices

    Notes
    -----
    This function handles both linescan and frame cameras. For linescan
    cameras, it returns multiple positions and rotations corresponding
    to different lines in the image. For frame cameras, it returns a
    single position and rotation.
    """
    # Read the first line from cam_file
    lineScan = isLinescan(cam_file)

    positions = []
    rotations = []

    if lineScan:
        # Read linescan data
        positions, rotations = read_linescan_pos_rot(cam_file)
    else:
        # read Pinhole (Frame) files in ASP .tsai or CSM .json format
        asp_dict = read_frame_cam_dict(cam_file)
        # get camera rotation
        position = asp_dict["cam_cen_ecef"]
        rot_mat = asp_dict["rotation_matrix"]
        positions.append(position)
        rotations.append(rot_mat)

    return (positions, rotations)


def read_positions_rotations(cams):
    """
    Read positions and rotations from multiple camera files.

    Parameters
    ----------
    cams : list of str
        List of paths to camera files

    Returns
    -------
    tuple
        Tuple containing (positions, rotations) where:
        - positions is a list of camera positions
        - rotations is a list of rotation matrices

    Raises
    ------
    SystemExit
        If the number of positions and rotations don't match

    Notes
    -----
    This function reads all the camera files and concatenates their
    positions and rotations into single lists.
    """
    positions, rotations = ([], [])
    for i in range(len(cams)):
        p, r = read_positions_rotations_from_file(cams[i])
        positions += p
        rotations += r

    # Must have as many rotations as positions. That is needed as later
    # we build ref rotations from positions.
    if len(rotations) != len(positions):
        print("Number of camera positions and orientations must be the same.")
        sys.exit(1)

    return (positions, rotations)


def read_angles(orig_cams, opt_cams, ref_cams):
    """
    Extract and convert camera orientations to roll, pitch, yaw angles.

    Parameters
    ----------
    orig_cams : list of str
        List of paths to original camera files
    opt_cams : list of str
        List of paths to optimized camera files
    ref_cams : list of str
        List of paths to reference camera files (can be empty)

    Returns
    -------
    tuple
        Tuple containing (orig_rotation_angles, opt_rotation_angles) where:
        - orig_rotation_angles is a list of Euler angles for original cameras
        - opt_rotation_angles is a list of Euler angles for optimized cameras

    Raises
    ------
    SystemExit
        If the number of original and reference cameras don't match

    Notes
    -----
    This function extracts the orientation of cameras and converts them
    to Euler angles (roll, pitch, yaw) relative to a reference orientation.
    If reference cameras are not provided, it estimates the reference
    orientation from the camera positions.
    """
    # orig_cams and ref_cams must be the same size
    if len(ref_cams) > 0 and len(orig_cams) != len(ref_cams):
        print(
            "Number of input and reference cameras must be the same. Got: ",
            len(ref_cams),
            " and ",
            len(opt_cams),
        )
        sys.exit(1)

    orig_positions, orig_rotations = read_positions_rotations(orig_cams)
    opt_positions, opt_rotations = read_positions_rotations(opt_cams)
    ref_positions, ref_rotations = read_positions_rotations(ref_cams)

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
