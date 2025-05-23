import json
import os
import sys

import contextily as ctx
import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Point

from asp_plot.utils import save_figure


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
        df = df.iloc[int(firstQuatIndex) : int(lastQuatIndex)]
        line_at_position = np.round(np.linspace(1, numLines, df.shape[0])).astype(int)
        df["line_at_position"] = line_at_position
    gdf = gpd.GeoDataFrame(df, geometry="original_positions")

    if map_crs:
        gdf.set_crs(epsg=map_crs, inplace=True)
    else:
        gdf.set_crs(epsg=4978, inplace=True)

    return gdf


def format_stat_value(value):
    """
    Format a numeric value with appropriate precision.

    Parameters
    ----------
    value : float
        The numeric value to format

    Returns
    -------
    str
        Formatted string representation of the value

    Notes
    -----
    Uses scientific notation for very small values (abs(value) < 0.01)
    and fixed-point notation with 2 decimal places otherwise.
    """
    return f"{value:.2e}" if abs(value) < 0.01 else f"{value:.2f}"


def plot_stats_text(ax, mean, std, unit="m"):
    """
    Add a text annotation showing statistics to a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to add the text
    mean : float
        Mean value to display
    std : float
        Standard deviation value to display
    unit : str, optional
        Unit to use for display, default is 'm'

    Notes
    -----
    The text is displayed in the lower left corner of the plot with
    a white background box for readability.
    """
    stats_text = f"{format_stat_value(mean)} ± {format_stat_value(std)} {unit}"
    ax.text(
        0.05,
        0.1,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def csm_camera_summary_plot(
    cam1_list,
    cam2_list=None,
    map_crs=None,
    title=None,
    trim=True,
    shared_scales=False,
    log_scale_positions=False,
    log_scale_angles=False,
    upper_magnitude_percentile=95,
    figsize=(20, 15),
    save_dir=None,
    fig_fn=None,
    add_basemap=False,
    **ctx_kwargs,
):
    """
    Generate a comprehensive summary plot comparing camera parameters.

    Parameters
    ----------
    cam1_list : list
        List containing paths to the original and optimized camera files for the first camera
    cam2_list : list or None, optional
        List containing paths to the original and optimized camera files for the second camera,
        default is None
    map_crs : int or None, optional
        EPSG code for the coordinate reference system for map plots, default is None (use ECEF)
    title : str or None, optional
        Additional title text to append to the default title, default is None
    trim : bool, optional
        Whether to trim data to image lines for linescan cameras, default is True
    shared_scales : bool, optional
        Whether to use the same y-axis scale for all position/angle plots, default is False
    log_scale_positions : bool, optional
        Whether to use logarithmic scale for position difference plots, default is False
    log_scale_angles : bool, optional
        Whether to use logarithmic scale for angle difference plots, default is False
    upper_magnitude_percentile : int, optional
        Percentile for setting colorbar upper limit, default is 95
    figsize : tuple, optional
        Figure size (width, height) in inches, default is (20, 15)
    save_dir : str or None, optional
        Directory to save the plot, default is None (don't save)
    fig_fn : str or None, optional
        Filename for the saved plot, default is None
    add_basemap : bool, optional
        Whether to add a basemap to map plots, default is False
    **ctx_kwargs
        Additional keyword arguments passed to contextily.add_basemap()

    Notes
    -----
    This function creates a comprehensive visualization comparing original and
    optimized camera parameters. It shows the spatial distribution of position
    and orientation differences, as well as detailed plots of individual
    components (x, y, z, roll, pitch, yaw). If two cameras are provided, the
    comparison is shown for both cameras in a larger figure.
    """

    original_camera1, optimized_camera1 = cam1_list
    cam1_name = os.path.basename(original_camera1).split(".")[0]
    gdf_cam1 = get_orbit_plot_gdf(
        original_camera1, optimized_camera1, map_crs=map_crs, trim=trim
    )

    if cam2_list:
        original_camera2, optimized_camera2 = cam2_list
        cam2_name = os.path.basename(original_camera2).split(".")[0]
        gdf_cam2 = get_orbit_plot_gdf(
            original_camera2,
            optimized_camera2,
            map_crs=map_crs,
            trim=trim,
        )

    if not map_crs and add_basemap:
        print(
            "\nWarning: Basemap will not be added to the plot because UTM map_crs is not provided.\n"
        )
        add_basemap = False

    # Calculate colorbar ranges
    position_values = gdf_cam1.position_diff_magnitude[
        gdf_cam1.position_diff_magnitude > 0
    ]
    angular_values = gdf_cam1.angular_diff_magnitude[
        gdf_cam1.angular_diff_magnitude > 0
    ]
    # When position or angular changes are all zero, the percentile calculation will fail
    # https://github.com/uw-cryo/asp_plot/issues/54
    try:
        cam1_position_vmin, cam1_position_vmax = np.percentile(
            position_values, [0, upper_magnitude_percentile]
        )
    except IndexError:
        cam1_position_vmin, cam1_position_vmax = 0, 0

    try:
        cam1_angular_vmin, cam1_angular_vmax = np.percentile(
            angular_values, [0, upper_magnitude_percentile]
        )
    except IndexError:
        cam1_angular_vmin, cam1_angular_vmax = 0, 0

    if upper_magnitude_percentile == 100:
        extend = "neither"
    else:
        extend = "max"

    # Calculate stats
    cam1_x_position_diff_mean, cam1_x_position_diff_std = (
        gdf_cam1.x_position_diff.mean(),
        gdf_cam1.x_position_diff.std(),
    )
    cam1_y_position_diff_mean, cam1_y_position_diff_std = (
        gdf_cam1.y_position_diff.mean(),
        gdf_cam1.y_position_diff.std(),
    )
    cam1_z_position_diff_mean, cam1_z_position_diff_std = (
        gdf_cam1.z_position_diff.mean(),
        gdf_cam1.z_position_diff.std(),
    )
    cam1_roll_diff_mean, cam1_roll_diff_std = (
        gdf_cam1.roll_diff.mean(),
        gdf_cam1.roll_diff.std(),
    )
    cam1_pitch_diff_mean, cam1_pitch_diff_std = (
        gdf_cam1.pitch_diff.mean(),
        gdf_cam1.pitch_diff.std(),
    )
    cam1_yaw_diff_mean, cam1_yaw_diff_std = (
        gdf_cam1.yaw_diff.mean(),
        gdf_cam1.yaw_diff.std(),
    )

    if cam2_list:
        cam2_x_position_diff_mean, cam2_x_position_diff_std = (
            gdf_cam2.x_position_diff.mean(),
            gdf_cam2.x_position_diff.std(),
        )
        cam2_y_position_diff_mean, cam2_y_position_diff_std = (
            gdf_cam2.y_position_diff.mean(),
            gdf_cam2.y_position_diff.std(),
        )
        cam2_z_position_diff_mean, cam2_z_position_diff_std = (
            gdf_cam2.z_position_diff.mean(),
            gdf_cam2.z_position_diff.std(),
        )
        cam2_roll_diff_mean, cam2_roll_diff_std = (
            gdf_cam2.roll_diff.mean(),
            gdf_cam2.roll_diff.std(),
        )
        cam2_pitch_diff_mean, cam2_pitch_diff_std = (
            gdf_cam2.pitch_diff.mean(),
            gdf_cam2.pitch_diff.std(),
        )
        cam2_yaw_diff_mean, cam2_yaw_diff_std = (
            gdf_cam2.yaw_diff.mean(),
            gdf_cam2.yaw_diff.std(),
        )

    # Begin plot
    if cam2_list:
        fig, axes = plt.subplots(4, 4, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(figsize[0], figsize[1] / 2))

    # Camera 1 mapview plot
    ax = axes[0, 0]
    gdf_cam1.plot(
        ax=ax,
        column="position_diff_magnitude",
        cmap="viridis",
        markersize=10,
        vmin=cam1_position_vmin,
        vmax=cam1_position_vmax,
    )
    ax.set_title(f"Camera 1: Position Change\n{cam1_name}", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    if add_basemap:
        ctx.add_basemap(ax=ax, **ctx_kwargs)
    sm1 = ScalarMappable(
        norm=Normalize(vmin=cam1_position_vmin, vmax=cam1_position_vmax), cmap="viridis"
    )
    cbar1 = plt.colorbar(
        sm1, ax=ax, extend=extend, orientation="vertical", aspect=30, pad=0.05
    )
    cbar1.set_label("Diff Magnitude (m)", fontsize=9)
    cbar1.ax.tick_params(labelsize=9)

    # Camera 1 angular mapview plot
    ax = axes[1, 0]
    gdf_cam1.plot(
        ax=ax,
        column="angular_diff_magnitude",
        cmap="inferno",
        markersize=10,
        vmin=cam1_angular_vmin,
        vmax=cam1_angular_vmax,
    )
    ax.set_title(f"Camera 1: Angle Change\n{cam1_name}", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    if add_basemap:
        ctx.add_basemap(ax=ax, **ctx_kwargs)
    sm2 = ScalarMappable(
        norm=Normalize(vmin=cam1_angular_vmin, vmax=cam1_angular_vmax), cmap="inferno"
    )
    cbar2 = plt.colorbar(
        sm2, ax=ax, extend=extend, orientation="vertical", aspect=30, pad=0.05
    )
    cbar2.set_label("Diff Magnitude (°)", fontsize=9)
    cbar2.ax.tick_params(labelsize=9)

    # Camera 1 line plots
    frame_cam1 = np.arange(gdf_cam1.shape[0])

    # Plot diffs in x, y, z for Camera 1
    ax1 = axes[0, 1]
    ax1.plot(
        frame_cam1,
        gdf_cam1.x_position_diff,
        c="#000080",
        lw=1,
        label="X position (easting)",
    )
    plot_stats_text(ax1, cam1_x_position_diff_mean, cam1_x_position_diff_std, unit="m")
    ax2 = axes[0, 2]
    ax2.plot(
        frame_cam1,
        gdf_cam1.y_position_diff,
        c="#4169E1",
        lw=1,
        label="Y position (northing)",
    )
    plot_stats_text(ax2, cam1_y_position_diff_mean, cam1_y_position_diff_std, unit="m")
    ax3 = axes[0, 3]
    ax3.plot(
        frame_cam1,
        gdf_cam1.z_position_diff,
        c="#87CEEB",
        lw=1,
        label="Z position (altitude)",
    )
    plot_stats_text(ax3, cam1_z_position_diff_mean, cam1_z_position_diff_std, unit="m")

    # Share y-axis for position diff plots
    min_val_position_diff = min(
        gdf_cam1.x_position_diff.min(),
        gdf_cam1.y_position_diff.min(),
        gdf_cam1.z_position_diff.min(),
    )
    max_val_position_diff = max(
        gdf_cam1.x_position_diff.max(),
        gdf_cam1.y_position_diff.max(),
        gdf_cam1.z_position_diff.max(),
    )

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(frame_cam1.min(), frame_cam1.max())
        ax.hlines(
            0, frame_cam1.min(), frame_cam1.max(), color="k", linestyle="-", lw=0.5
        )
        ax.set_title("Camera 1", loc="right", fontsize=10, y=0.98)
        if "line_at_position" in gdf_cam1.columns:
            ax.set_xlabel("Linescan Image Line Number", fontsize=9)
            xticks = ax.get_xticks()
            xticks = xticks[(xticks >= frame_cam1.min()) & (xticks <= frame_cam1.max())]
            xtick_labels = [
                (
                    gdf_cam1.iloc[int(x)].line_at_position
                    if int(x) < len(gdf_cam1.line_at_position)
                    else ""
                )
                for x in xticks
            ]
            xtick_labels = [np.round(x, -1) for x in xtick_labels]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
        else:
            ax.set_xlabel("Position Sample", fontsize=9)
        ax.set_ylabel("Original $-$ Optimized (m)", fontsize=9)
        if shared_scales:
            ax.set_ylim(min_val_position_diff, max_val_position_diff)
        if log_scale_positions:
            ax.set_yscale("symlog")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)

    # Plot diffs in roll, pitch, yaw for Camera 1
    ax1 = axes[1, 1]
    ax1.plot(frame_cam1, gdf_cam1.roll_diff, c="#FF4500", lw=1, label="Roll Diff")
    ax1_r = ax1.twinx()
    ax1_r.plot(
        frame_cam1,
        gdf_cam1.original_roll,
        c="k",
        lw=1,
        linestyle="--",
        label="Original Roll",
    )
    plot_stats_text(ax1_r, cam1_roll_diff_mean, cam1_roll_diff_std, unit="°")

    ax2 = axes[1, 2]
    ax2.plot(frame_cam1, gdf_cam1.pitch_diff, c="#FFA500", lw=1, label="Pitch Diff")
    ax2_r = ax2.twinx()
    ax2_r.plot(
        frame_cam1,
        gdf_cam1.original_pitch,
        c="k",
        lw=1,
        linestyle="--",
        label="Original Pitch",
    )
    plot_stats_text(ax2_r, cam1_pitch_diff_mean, cam1_pitch_diff_std, unit="°")

    ax3 = axes[1, 3]
    ax3.plot(frame_cam1, gdf_cam1.yaw_diff, c="#FFB347", lw=1, label="Yaw Diff")
    ax3_r = ax3.twinx()
    ax3_r.plot(
        frame_cam1,
        gdf_cam1.original_yaw,
        c="k",
        lw=1,
        linestyle="--",
        label="Original Yaw",
    )
    plot_stats_text(ax3_r, cam1_yaw_diff_mean, cam1_yaw_diff_std, unit="°")

    # Share y-axis for angular diff plots
    min_val_angle_diff = min(
        gdf_cam1.roll_diff.min(), gdf_cam1.pitch_diff.min(), gdf_cam1.yaw_diff.min()
    )
    max_val_angle_diff = max(
        gdf_cam1.roll_diff.max(), gdf_cam1.pitch_diff.max(), gdf_cam1.yaw_diff.max()
    )

    for ax, ax_r in [(ax1, ax1_r), (ax2, ax2_r), (ax3, ax3_r)]:
        ax.set_xlim(frame_cam1.min(), frame_cam1.max())
        ax.hlines(
            0, frame_cam1.min(), frame_cam1.max(), color="k", linestyle="-", lw=0.5
        )
        ax.set_title("Camera 1", loc="right", fontsize=10, y=0.98)
        if "line_at_position" in gdf_cam1.columns:
            ax.set_xlabel("Linescan Image Line Number", fontsize=9)
            xticks = ax.get_xticks()
            xticks = xticks[(xticks >= frame_cam1.min()) & (xticks <= frame_cam1.max())]
            xtick_labels = [
                (
                    gdf_cam1.iloc[int(x)].line_at_position
                    if int(x) < len(gdf_cam1.line_at_position)
                    else ""
                )
                for x in xticks
            ]
            xtick_labels = [np.round(x, -1) for x in xtick_labels]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xtick_labels)
        else:
            ax.set_xlabel("Position Sample", fontsize=9)
        ax.set_ylabel("Original $-$ Optimized (°)", fontsize=9)
        if shared_scales:
            ax.set_ylim(min_val_angle_diff, max_val_angle_diff)
        ax_r.set_ylabel("Original (°)", fontsize=9)
        if log_scale_angles:
            ax.set_yscale("symlog")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax_r.tick_params(axis="both", which="major", labelsize=9)
        lines1, labels1 = ax_r.get_legend_handles_labels()
        lines2, labels2 = ax.get_legend_handles_labels()
        ax_r.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Camera 2 mapview plot
    if cam2_list:
        ax = axes[2, 0]
        gdf_cam2.plot(
            ax=ax,
            column="position_diff_magnitude",
            cmap="viridis",
            markersize=10,
            vmin=cam1_position_vmin,
            vmax=cam1_position_vmax,
        )
        ax.set_title(f"Camera 2: Position Change\n{cam2_name}", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        if add_basemap:
            ctx.add_basemap(ax=ax, **ctx_kwargs)
        sm1 = ScalarMappable(
            norm=Normalize(vmin=cam1_position_vmin, vmax=cam1_position_vmax),
            cmap="viridis",
        )
        cbar1 = plt.colorbar(
            sm1, ax=ax, extend=extend, orientation="vertical", aspect=30, pad=0.05
        )
        cbar1.set_label("Diff Magnitude (m)", fontsize=9)
        cbar1.ax.tick_params(labelsize=9)

        # Camera 2 angular mapview plot
        ax = axes[3, 0]
        gdf_cam2.plot(
            ax=ax,
            column="angular_diff_magnitude",
            cmap="inferno",
            markersize=10,
            vmin=cam1_angular_vmin,
            vmax=cam1_angular_vmax,
        )
        ax.set_title(f"Camera 2: Angle Change\n{cam2_name}", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        if add_basemap:
            ctx.add_basemap(ax=ax, **ctx_kwargs)
        sm2 = ScalarMappable(
            norm=Normalize(vmin=cam1_angular_vmin, vmax=cam1_angular_vmax),
            cmap="inferno",
        )
        cbar2 = plt.colorbar(
            sm2, ax=ax, extend=extend, orientation="vertical", aspect=30, pad=0.05
        )
        cbar2.set_label("Diff Magnitude (°)", fontsize=9)
        cbar2.ax.tick_params(labelsize=9)

        frame_cam2 = np.arange(gdf_cam2.shape[0])

        # Plot diffs in x, y, z for Camera 2
        ax1 = axes[2, 1]
        ax1.plot(
            frame_cam2,
            gdf_cam2.x_position_diff,
            c="#000080",
            lw=1,
            label="X position (easting)",
        )
        plot_stats_text(
            ax1, cam2_x_position_diff_mean, cam2_x_position_diff_std, unit="m"
        )
        ax2 = axes[2, 2]
        ax2.plot(
            frame_cam2,
            gdf_cam2.y_position_diff,
            c="#4169E1",
            lw=1,
            label="Y position (northing)",
        )
        plot_stats_text(
            ax2, cam2_y_position_diff_mean, cam2_y_position_diff_std, unit="m"
        )
        ax3 = axes[2, 3]
        ax3.plot(
            frame_cam2,
            gdf_cam2.z_position_diff,
            c="#87CEEB",
            lw=1,
            label="Z position (altitude)",
        )
        plot_stats_text(
            ax3, cam2_z_position_diff_mean, cam2_z_position_diff_std, unit="m"
        )

        # Share y-axis for position diff plots
        min_val_position_diff = min(
            gdf_cam2.x_position_diff.min(),
            gdf_cam2.y_position_diff.min(),
            gdf_cam2.z_position_diff.min(),
        )
        max_val_position_diff = max(
            gdf_cam2.x_position_diff.max(),
            gdf_cam2.y_position_diff.max(),
            gdf_cam2.z_position_diff.max(),
        )

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(frame_cam2.min(), frame_cam2.max())
            ax.hlines(
                0, frame_cam2.min(), frame_cam2.max(), color="k", linestyle="-", lw=0.5
            )
            ax.set_title("Camera 2", loc="right", fontsize=10, y=0.98)
            if "line_at_position" in gdf_cam2.columns:
                ax.set_xlabel("Linescan Image Line Number", fontsize=9)
                xticks = ax.get_xticks()
                xticks = xticks[
                    (xticks >= frame_cam2.min()) & (xticks <= frame_cam2.max())
                ]
                xtick_labels = [
                    (
                        gdf_cam2.iloc[int(x)].line_at_position
                        if int(x) < len(gdf_cam2.line_at_position)
                        else ""
                    )
                    for x in xticks
                ]
                xtick_labels = [np.round(x, -1) for x in xtick_labels]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels)
            else:
                ax.set_xlabel("Position Sample", fontsize=9)
            ax.set_ylabel("Original $-$ Optimized (m)", fontsize=9)
            if shared_scales:
                ax.set_ylim(min_val_position_diff, max_val_position_diff)
            if log_scale_positions:
                ax.set_yscale("symlog")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
            ax.legend(loc="upper right", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=8)

        # Plot diffs in roll, pitch, yaw for Camera 2
        ax1 = axes[3, 1]
        ax1.plot(frame_cam2, gdf_cam2.roll_diff, c="#FF4500", lw=1, label="Roll Diff")
        ax1_r = ax1.twinx()
        ax1_r.plot(
            frame_cam2,
            gdf_cam2.original_roll,
            c="k",
            lw=1,
            linestyle="--",
            label="Original Roll",
        )
        plot_stats_text(ax1_r, cam2_roll_diff_mean, cam2_roll_diff_std, unit="°")

        ax2 = axes[3, 2]
        ax2.plot(frame_cam2, gdf_cam2.pitch_diff, c="#FFA500", lw=1, label="Pitch Diff")
        ax2_r = ax2.twinx()
        ax2_r.plot(
            frame_cam2,
            gdf_cam2.original_pitch,
            c="k",
            lw=1,
            linestyle="--",
            label="Original Pitch",
        )
        plot_stats_text(ax2_r, cam2_pitch_diff_mean, cam2_pitch_diff_std, unit="°")

        ax3 = axes[3, 3]
        ax3.plot(frame_cam2, gdf_cam2.yaw_diff, c="#FFB347", lw=1, label="Yaw Diff")
        ax3_r = ax3.twinx()
        ax3_r.plot(
            frame_cam2,
            gdf_cam2.original_yaw,
            c="k",
            lw=1,
            linestyle="--",
            label="Original Yaw",
        )
        plot_stats_text(ax3_r, cam2_yaw_diff_mean, cam2_yaw_diff_std, unit="°")

        # Share y-axis for angular diff plots
        min_val_angle_diff = min(
            gdf_cam2.roll_diff.min(), gdf_cam2.pitch_diff.min(), gdf_cam2.yaw_diff.min()
        )
        max_val_angle_diff = max(
            gdf_cam2.roll_diff.max(), gdf_cam2.pitch_diff.max(), gdf_cam2.yaw_diff.max()
        )

        for ax, ax_r in [(ax1, ax1_r), (ax2, ax2_r), (ax3, ax3_r)]:
            ax.set_xlim(frame_cam2.min(), frame_cam2.max())
            ax.hlines(
                0, frame_cam2.min(), frame_cam2.max(), color="k", linestyle="-", lw=0.5
            )
            ax.set_title("Camera 2", loc="right", fontsize=10, y=0.98)
            if "line_at_position" in gdf_cam2.columns:
                ax.set_xlabel("Linescan Image Line Number", fontsize=9)
                xticks = ax.get_xticks()
                xticks = xticks[
                    (xticks >= frame_cam2.min()) & (xticks <= frame_cam2.max())
                ]
                xtick_labels = [
                    (
                        gdf_cam2.iloc[int(x)].line_at_position
                        if int(x) < len(gdf_cam2.line_at_position)
                        else ""
                    )
                    for x in xticks
                ]
                xtick_labels = [np.round(x, -1) for x in xtick_labels]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels)
            else:
                ax.set_xlabel("Position Sample", fontsize=9)
            ax.set_ylabel("Original $-$ Optimized (°)", fontsize=9)
            if shared_scales:
                ax.set_ylim(min_val_angle_diff, max_val_angle_diff)
            ax_r.set_ylabel("Original (°)", fontsize=9)
            if log_scale_angles:
                ax.set_yscale("symlog")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax_r.tick_params(axis="both", which="major", labelsize=9)
            lines1, labels1 = ax_r.get_legend_handles_labels()
            lines2, labels2 = ax.get_legend_handles_labels()
            ax_r.legend(
                lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8
            )

        # Set linewidth and color for all spines
        for ax in axes[:2].flatten():
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color("#141414")

        for ax in axes[2:].flatten():
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color("#A9A9A9")

    title_text = f"{'{}: '.format(title) if title else ''}Position and Angle Changes for Camera 1 ({cam1_name}){' and Camera 2 ({})'.format(cam2_name) if cam2_list else ''}"

    if map_crs:
        title_text += (
            f"\n(original positions in ECEF, projected here to UTM EPSG:{map_crs})"
        )
    else:
        title_text += "\n(original positions in ECEF)"

    fig.suptitle(
        title_text,
        fontsize=12,
    )

    plt.tight_layout()
    if save_dir and fig_fn:
        save_figure(fig, save_dir, fig_fn)

    plt.show()


#
# Methods below copied from orbit_plot.py in the ASP source code:
# https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/orbit_plot.py
#

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
        (positions, rotations) = read_linescan_pos_rot(cam_file)
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
