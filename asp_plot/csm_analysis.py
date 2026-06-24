"""
Camera-model analysis: turn original/optimized camera pairs into the
position- and orientation-difference GeoDataFrame consumed by the plotting
layer (``csm_camera.py``).

This module owns the asp_plot-specific analysis (``get_orbit_plot_gdf``,
``reproject_ecef``, ``poly_fit``) and builds on the ASP-mirrored readers in
``csm_io.py``.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from asp_plot.csm_io import (
    getTimeAtLine,
    isLinescan,
    read_angles,
    read_csm_cam,
    read_positions_rotations,
)


def reproject_ecef(positions, to_epsg=4326):
    """
    Reproject ECEF coordinates to a specified EPSG coordinate system.

    Parameters
    ----------
    positions : numpy.ndarray
        A 2D array of ECEF coordinates, where each row represents a point
    to_epsg : int, optional
        The EPSG code of the target coordinate system, default is 4326 (WGS84)

    Returns
    -------
    numpy.ndarray
        A 2D array of reprojected coordinates in the target EPSG coordinate system

    Notes
    -----
    ECEF (Earth-Centered, Earth-Fixed) coordinates are a 3D Cartesian coordinate
    system with the origin at the center of the Earth. This function converts
    those coordinates to a different coordinate system specified by an EPSG code.
    """
    transformer = Transformer.from_crs("EPSG:4978", f"EPSG:{to_epsg}")
    x, y, z = transformer.transform(positions[:, 0], positions[:, 1], positions[:, 2])
    return np.column_stack((x, y, z))


def get_orbit_plot_gdf(original_camera, optimized_camera, map_crs=None, trim=True):
    """
    Create a GeoDataFrame containing camera positions and orientation differences.

    Parameters
    ----------
    original_camera : str
        Path to the original camera file
    optimized_camera : str
        Path to the optimized camera file
    map_crs : int or None, optional
        EPSG code for the target coordinate system, default is None (keep ECEF)
    trim : bool, optional
        Whether to trim data to only the first and last image lines for linescan
        cameras, default is True

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing camera positions and orientation data with
        columns for position differences, angle differences, and original values

    Notes
    -----
    This function compares the original and optimized camera models and
    calculates the differences in position and orientation. For linescan
    cameras, it optionally trims the data to only include samples corresponding
    to the actual image lines.
    """
    # orbit_plot.py method to get angles in NED
    # https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/orbit_plot.py#L412
    # This method already calls read_positions_rotations below, but it
    # doesn't return the positions and rotations we want for plotting
    original_rotation_angles, optimized_rotation_angles = read_angles(
        [original_camera], [optimized_camera], []
    )

    # orbit_plot.py method to get positions and rotations
    # https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/orbit_plot.py#L381
    # Could be retrieved from the above method, by adding to return statement there
    original_positions_ecef, original_rotations = read_positions_rotations(
        [original_camera]
    )
    optimized_positions_ecef, optimized_rotations = read_positions_rotations(
        [optimized_camera]
    )
    original_positions_ecef = np.array(original_positions_ecef)
    optimized_positions_ecef = np.array(optimized_positions_ecef)

    if trim and isLinescan(optimized_camera):
        # Find the pose indices for the first and last image lines
        j = read_csm_cam(optimized_camera)
        t0 = j["m_t0Quat"]
        dt = j["m_dtQuat"]
        numLines = j["m_nLines"]
        firstLineTime = getTimeAtLine(j, 0)
        firstQuatIndex = int(round((firstLineTime - t0) / dt))
        lastLineTime = getTimeAtLine(j, numLines - 1)
        lastQuatIndex = int(round((lastLineTime - t0) / dt))

        # To get the first line and last image line:
        # firstLine = getLineAtTime(firstLineTime - t0, j)
        # lastLine = getLineAtTime(lastLineTime - t0, j)
        # Or done below with simple interpolation to get line_at_position
        # since we know this must follow a linear relationship
    if not isLinescan(optimized_camera):
        print(
            "Warning: Camera model is not linescan. Cannot trim to first and last image lines."
        )

    if len(original_positions_ecef) != len(optimized_positions_ecef):
        original_positions_ecef = np.array(
            [
                np.interp(
                    np.linspace(0, 1, len(optimized_positions_ecef)),
                    np.linspace(0, 1, len(original_positions_ecef)),
                    original_positions_ecef[:, i],
                )
                for i in range(3)
            ]
        ).T

    # Taken directly from orbit_plot.py
    # https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/orbit_plot.py#L600-L607
    # "The order is roll, pitch, yaw, as returned by R.from_matrix().as_euler('XYZ',degrees=True)"
    original_roll = np.array([r[0] for r in original_rotation_angles])
    original_pitch = np.array([r[1] for r in original_rotation_angles])
    original_yaw = np.array([r[2] for r in original_rotation_angles])
    optimized_roll = np.array([r[0] for r in optimized_rotation_angles])
    optimized_pitch = np.array([r[1] for r in optimized_rotation_angles])
    optimized_yaw = np.array([r[2] for r in optimized_rotation_angles])

    # Interpolate original angles if lengths don't match
    if len(original_roll) != len(optimized_roll):
        original_roll = np.interp(
            np.linspace(0, 1, len(optimized_roll)),
            np.linspace(0, 1, len(original_roll)),
            original_roll,
        )
        original_pitch = np.interp(
            np.linspace(0, 1, len(optimized_pitch)),
            np.linspace(0, 1, len(original_pitch)),
            original_pitch,
        )
        original_yaw = np.interp(
            np.linspace(0, 1, len(optimized_yaw)),
            np.linspace(0, 1, len(original_yaw)),
            original_yaw,
        )

    # We are interested in the difference between the original and optimized angles
    roll_diff = original_roll - optimized_roll
    pitch_diff = original_pitch - optimized_pitch
    yaw_diff = original_yaw - optimized_yaw

    # Also get angular diff magnitude
    angular_diff_magnitudes = np.sqrt(roll_diff**2 + pitch_diff**2 + yaw_diff**2)

    # Reproject positions from ECEF to map_crs
    if map_crs:
        original_positions = reproject_ecef(original_positions_ecef, to_epsg=map_crs)
        optimized_positions = reproject_ecef(optimized_positions_ecef, to_epsg=map_crs)
    else:
        original_positions = original_positions_ecef
        optimized_positions = optimized_positions_ecef

    # Calculate the difference between the original and optimized positions
    position_diffs = original_positions - optimized_positions
    x_position_diff = position_diffs[:, 0]
    y_position_diff = position_diffs[:, 1]
    z_position_diff = position_diffs[:, 2]

    # Get the magntiude of position difference
    # Below is equivalent to: np.sqrt(x_position_diff**2 + y_position_diff**2 + z_position_diff**2)
    position_diff_magnitudes = np.linalg.norm(position_diffs, axis=1)

    # Build a GeoDataFrame for plotting
    data = {
        "original_positions": [Point(x, y, z) for x, y, z in original_positions],
        "position_diff_magnitude": position_diff_magnitudes,
        "x_position_diff": x_position_diff,
        "y_position_diff": y_position_diff,
        "z_position_diff": z_position_diff,
        "angular_diff_magnitude": angular_diff_magnitudes,
        "original_roll": original_roll,
        "original_pitch": original_pitch,
        "original_yaw": original_yaw,
        "optimized_roll": optimized_roll,
        "optimized_pitch": optimized_pitch,
        "optimized_yaw": optimized_yaw,
        "roll_diff": roll_diff,
        "pitch_diff": pitch_diff,
        "yaw_diff": yaw_diff,
    }
    df = pd.DataFrame(data)
    if trim and isLinescan(optimized_camera):
        if firstQuatIndex < lastQuatIndex:
            df = df.iloc[int(firstQuatIndex) : int(lastQuatIndex)]
        else:
            df = df.iloc[int(lastQuatIndex) : int(firstQuatIndex)]
        line_at_position = np.round(np.linspace(1, numLines, df.shape[0])).astype(int)
        df["line_at_position"] = line_at_position
    gdf = gpd.GeoDataFrame(df, geometry="original_positions")

    if map_crs:
        gdf.set_crs(epsg=map_crs, inplace=True)
    else:
        gdf.set_crs(epsg=4978, inplace=True)

    return gdf


def poly_fit(X, Y):
    """
    Fit a linear polynomial to data and return the fitted values.

    Parameters
    ----------
    X : array-like
        Independent variable values
    Y : array-like
        Dependent variable values

    Returns
    -------
    numpy.ndarray
        Fitted Y values from a degree 1 polynomial fit
    """
    fit = np.poly1d(np.polyfit(X, Y, 1))
    return fit(X)
