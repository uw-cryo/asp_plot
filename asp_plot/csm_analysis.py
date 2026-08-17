"""
Camera-model analysis: turn original/optimized camera pairs into the
position- and orientation-difference GeoDataFrame consumed by the plotting
layer (``csm_camera.py``).

This module owns the asp_plot-specific analysis (``get_orbit_plot_gdf``,
``read_angles_common_frame``, ``wrap_angle_diff``, ``reproject_ecef``,
``poly_fit``) and builds on the ASP-mirrored readers in ``csm_io.py``.

``read_angles_common_frame`` is a deliberate divergence from ASP's
``orbit_plot.py``: differencing two cameras requires them to share one reference
frame, and ASP estimates a separate frame per camera from that camera's own
ephemeris. See its docstring and issue #53.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

from asp_plot.csm_io import (
    estim_satellite_orientation,
    getTimeAtLine,
    isLinescan,
    read_angles,
    read_csm_cam,
    read_positions_rotations,
    roll_pitch_yaw,
)


def _resample(values, n_out):
    """Resample an (n_in, k) array onto ``n_out`` samples spanning the same range."""
    values = np.asarray(values, dtype=float)
    n_in = values.shape[0]
    if n_in == n_out:
        return values
    x_out = np.linspace(0, 1, n_out)
    x_in = np.linspace(0, 1, n_in)
    return np.column_stack(
        [np.interp(x_out, x_in, values[:, i]) for i in range(values.shape[1])]
    )


def wrap_angle_diff(angles):
    """
    Wrap angle differences in degrees into the [-180, 180) range.

    Parameters
    ----------
    angles : array-like
        Angle differences in degrees

    Returns
    -------
    numpy.ndarray
        The same differences, wrapped into [-180, 180)

    Notes
    -----
    Euler angles are recovered on a branch cut, so a camera whose yaw sits near
    +/-180 degrees (a backward-looking sensor such as ASTER's 3B band, for
    example) can have adjacent samples reported as +179.9 and -179.9. Without
    wrapping, an orientation change of 0.2 degrees is plotted as ~360 degrees.
    """
    return (np.asarray(angles, dtype=float) + 180.0) % 360.0 - 180.0


def read_angles_common_frame(original_camera, optimized_camera):
    """
    Read roll/pitch/yaw for a camera pair using a single shared reference frame.

    Parameters
    ----------
    original_camera : str
        Path to the original camera file
    optimized_camera : str
        Path to the optimized camera file

    Returns
    -------
    tuple of numpy.ndarray
        ``(original_angles, optimized_angles)``, each of shape (n, 3) holding
        roll, pitch and yaw in degrees

    Notes
    -----
    ASP's ``orbit_plot.py`` (mirrored in ``csm_io.read_angles()``) estimates the
    satellite body frame separately for each camera, from a central difference
    of that camera's own ephemeris. That is fine when plotting one camera, but
    it corrupts a *difference* between two cameras: ``bundle_adjust`` and
    ``jitter_solve`` both perturb the positions and usually resample the
    ephemeris to a finer spacing, and a small position perturbation over a short
    baseline tilts the estimated along-track axis a lot. At WorldView's ~7 km/s
    and a 0.01 s sample spacing the baseline is only ~140 m, so a 2 m radial
    perturbation swings the estimated frame by ~0.8 degrees -- orders of
    magnitude more than the orientation change actually being measured, and it
    lands almost entirely in pitch.

    Here both cameras are instead expressed in one frame, estimated from the
    original (unperturbed) ephemeris and resampled onto the optimized camera's
    sample grid. The reported angle difference is then the true relative
    rotation between the two camera models, not a difference of two different
    reference frames.
    """
    original_positions, original_rotations = read_positions_rotations([original_camera])
    optimized_positions, optimized_rotations = read_positions_rotations(
        [optimized_camera]
    )
    original_positions = np.array(original_positions, dtype=float)

    # A single-sample (frame) camera gives no baseline to estimate a satellite
    # frame from, so fall back to ASP's behavior. Guard on *both* cameras: a
    # one-sample optimized camera would collapse the resample below to a single
    # point, which has the same zero-length tangent vector problem. Neither path
    # produces usable angles for a frame camera -- ASP's own
    # estim_satellite_orientation divides by zero there too -- so this only
    # keeps the two cameras treated alike.
    if len(original_positions) < 2 or len(optimized_rotations) < 2:
        original_angles, optimized_angles = read_angles(
            [original_camera], [optimized_camera], []
        )
        return np.array(original_angles), np.array(optimized_angles)

    original_ref_rotations = estim_satellite_orientation(original_positions)
    optimized_ref_rotations = estim_satellite_orientation(
        _resample(original_positions, len(optimized_rotations))
    )

    original_angles = np.array(
        [
            roll_pitch_yaw(original_rotations[i], original_ref_rotations[i])
            for i in range(len(original_rotations))
        ]
    )
    optimized_angles = np.array(
        [
            roll_pitch_yaw(optimized_rotations[i], optimized_ref_rotations[i])
            for i in range(len(optimized_rotations))
        ]
    )
    return original_angles, optimized_angles


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
    # Roll/pitch/yaw for both cameras, expressed in a single satellite frame
    # estimated from the original ephemeris. This follows orbit_plot.py's
    # read_angles() (mirrored in csm_io) but shares one reference frame between
    # the two cameras, so the difference below is not contaminated by the
    # frame itself moving. See read_angles_common_frame() for why.
    original_rotation_angles, optimized_rotation_angles = read_angles_common_frame(
        original_camera, optimized_camera
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

    # Interpolate original angles if lengths don't match. Unwrap first so a
    # series straddling the +/-180 branch cut is not averaged through zero,
    # then wrap the result back into [-180, 180).
    if len(original_roll) != len(optimized_roll):
        original_angles = _resample(
            np.unwrap(
                np.column_stack([original_roll, original_pitch, original_yaw]),
                period=360.0,
                axis=0,
            ),
            len(optimized_roll),
        )
        original_roll, original_pitch, original_yaw = wrap_angle_diff(original_angles).T

    # We are interested in the difference between the original and optimized
    # angles. Wrap the differences so a camera pointing near +/-180 degrees in
    # yaw does not report a ~360 degree change across the branch cut.
    roll_diff = wrap_angle_diff(original_roll - optimized_roll)
    pitch_diff = wrap_angle_diff(original_pitch - optimized_pitch)
    yaw_diff = wrap_angle_diff(original_yaw - optimized_yaw)

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
