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
    Reproject a set of ECEF (Earth-Centered, Earth-Fixed) coordinates to a specified EPSG coordinate system.

    Args:
        positions (numpy.ndarray): A 2D array of ECEF coordinates, where each row represents a point.
        to_epsg (int): The EPSG code of the target coordinate system to reproject to. Defaults to 4326 (WGS84).

    Returns:
        numpy.ndarray: A 2D array of reprojected coordinates in the target EPSG coordinate system.
    """
    transformer = Transformer.from_crs("EPSG:4978", f"EPSG:{to_epsg}")
    x, y, z = transformer.transform(positions[:, 0], positions[:, 1], positions[:, 2])
    return np.column_stack((x, y, z))


def get_orbit_plot_gdf(original_camera, optimized_camera, map_crs=None):
    """
    Get a GeoDataFrame containing the original and optimized camera positions, as well as the differences between them.

    Args:
        original_camera (list): A list containing the original cameras.
        optimized_camera (list): A list containing the optimized cameras.
        map_crs (int, optional): The EPSG code of the target coordinate system to reproject the camera positions to. If not provided, the positions will be returned in ECEF coordinates.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the following columns:
            - original_positions: A shapely.geometry.Point for each original camera position.
            - position_diff_magnitude: The magnitude of the difference between the original and optimized camera positions.
            - x_position_diff: The difference in the x-coordinate between the original and optimized camera positions.
            - y_position_diff: The difference in the y-coordinate between the original and optimized camera positions.
            - z_position_diff: The difference in the z-coordinate between the original and optimized camera positions.
            - angular_diff_magnitude: The magnitude of the difference between the original and optimized camera rotation angles.
            - original_roll: The roll angle of the original camera.
            - original_pitch: The pitch angle of the original camera.
            - original_yaw: The yaw angle of the original camera.
            - optimized_roll: The roll angle of the optimized camera.
            - optimized_pitch: The pitch angle of the optimized camera.
            - optimized_yaw: The yaw angle of the optimized camera.
            - roll_diff: The difference in roll angle between the original and optimized cameras.
            - pitch_diff: The difference in pitch angle between the original and optimized cameras.
            - yaw_diff: The difference in yaw angle between the original and optimized cameras.
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

    # Interpolate original values if lengths don't match
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
    gdf = gpd.GeoDataFrame(df, geometry="original_positions")

    if map_crs:
        gdf.set_crs(epsg=map_crs, inplace=True)
    else:
        gdf.set_crs(epsg=4978, inplace=True)

    return gdf


def trim_gdf(gdf, near_zero_tolerance=1e-8, trim_percentage=10):
    """
    Trims a GeoDataFrame by removing the first and last entries that have a position difference magnitude close to zero.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to be trimmed.
        near_zero_tolerance (float, optional): The tolerance value for considering a position difference magnitude as close to zero. Defaults to 1e-8.
        trim_percentage (float, optional): Any additional percentage of the total length to trim from the start and end. Defaults to 10.

    Returns:
        gpd.GeoDataFrame: The trimmed GeoDataFrame.
    """
    non_zero_indices = np.where(
        np.abs(gdf.position_diff_magnitude) > near_zero_tolerance
    )[0]
    # Find the first non-zero value from the start
    start_index = non_zero_indices[0]
    # Find the first non-zero value from the end
    end_index = non_zero_indices[-1]
    # Apply additional trimming
    total_length = end_index - start_index + 1
    additional_trim = int(total_length * (trim_percentage / 100) / 2)
    start_index += additional_trim
    end_index -= additional_trim

    return gdf.iloc[start_index : end_index + 1].reset_index(drop=True)


def format_stat_value(value):
    """
    Formats a numeric value as a string with appropriate precision.

    If the absolute value of the input value is less than 0.01, the value is formatted using scientific notation with 2 decimal places. Otherwise, the value is formatted as a fixed-point number with 2 decimal places.

    Args:
        value (float): The numeric value to be formatted.

    Returns:
        str: The formatted string representation of the input value.
    """
    return f"{value:.2e}" if abs(value) < 0.01 else f"{value:.2f}"


def plot_stats_text(ax, mean, std):
    """
    Plots a text annotation on the given axis displaying the mean and standard deviation of a value.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the text annotation.
        mean (float): The mean value to display.
        std (float): The standard deviation value to display.

    Returns:
        None
    """
    stats_text = f"{format_stat_value(mean)} ± {format_stat_value(std)} m"
    ax.text(
        0.05,
        0.1,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )


def summary_plot_two_camera_optimization(
    cam1_list,
    cam2_list,
    map_crs=None,
    title=None,
    trim=False,
    near_zero_tolerance=1e-3,
    trim_percentage=5,
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
    Generates a summary plot comparing the position and angle changes between the original and optimized camera parameters for two cameras.

    Args:
        cam1_list (list): A list containing the original and optimized camera files for the first camera.
        cam2_list (list): A list containing the original and optimized camera files for the second camera.
        map_crs (int, optional): The EPSG code for the coordinate reference system to use for the map plots. If not provided, the plots will use the original ECEF coordinates.
        title (str, optional): An additional descriptive title to append to the overall plot title containing the camera names.
        trim (bool, optional): Whether to trim the beginning and end of the data to remove near-zero values.
        near_zero_tolerance (float, optional): The tolerance value for considering a value to be near zero if `trim` is True.
        trim_percentage (float, optional): The additional percentage of data to trim from the beginning and end if `trim` is True.
        shared_scales (bool, optional): Whether to use shared y-axis scales for the position and angle difference plots.
        log_scale_positions (bool, optional): Whether to use a logarithmic scale for the position difference plots.
        log_scale_angles (bool, optional): Whether to use a logarithmic scale for the angle difference plots.
        upper_magnitude_percentile (int, optional): The upper percentile to use for the colorbar ranges for the mapview plots.
        figsize (tuple, optional): The size of the figure to generate.
        save_dir (str, optional): The directory to save the generated figure to.
        fig_fn (str, optional): The filename to use for the saved figure.
        add_basemap (bool, optional): Whether to add a basemap to the map plots.
        **ctx_kwargs: Additional keyword arguments to pass to the `ctx.add_basemap` function.

    Returns:
        None
    """

    original_camera1, optimized_camera1 = cam1_list
    original_camera2, optimized_camera2 = cam2_list
    cam1_name = os.path.basename(original_camera1).split(".")[0]
    cam2_name = os.path.basename(original_camera2).split(".")[0]
    gdf_cam1 = get_orbit_plot_gdf(original_camera1, optimized_camera1, map_crs=map_crs)
    gdf_cam2 = get_orbit_plot_gdf(original_camera2, optimized_camera2, map_crs=map_crs)

    if not map_crs and add_basemap:
        print(
            "\nWarning: Basemap will not be added to the plot because UTM map_crs is not provided.\n"
        )
        add_basemap = False

    # Trim the beginning and end of the geodataframes
    if trim:
        gdf_cam1 = trim_gdf(
            gdf_cam1,
            near_zero_tolerance=near_zero_tolerance,
            trim_percentage=trim_percentage,
        )
        gdf_cam2 = trim_gdf(
            gdf_cam2,
            near_zero_tolerance=near_zero_tolerance,
            trim_percentage=trim_percentage,
        )

    # Calculate colorbar ranges
    position_values = gdf_cam1.position_diff_magnitude[
        gdf_cam1.position_diff_magnitude > 0
    ]
    angular_values = gdf_cam1.angular_diff_magnitude[
        gdf_cam1.angular_diff_magnitude > 0
    ]
    cam1_position_vmin, cam1_position_vmax = np.percentile(
        position_values, [0, upper_magnitude_percentile]
    )
    cam1_angular_vmin, cam1_angular_vmax = np.percentile(
        angular_values, [0, upper_magnitude_percentile]
    )
    position_values = gdf_cam2.position_diff_magnitude[
        gdf_cam2.position_diff_magnitude > 0
    ]
    angular_values = gdf_cam2.angular_diff_magnitude[
        gdf_cam2.angular_diff_magnitude > 0
    ]
    cam2_position_vmin, cam2_position_vmax = np.percentile(
        position_values, [0, upper_magnitude_percentile]
    )
    cam2_angular_vmin, cam2_angular_vmax = np.percentile(
        angular_values, [0, upper_magnitude_percentile]
    )

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
    fig, axes = plt.subplots(4, 4, figsize=figsize)

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
    cbar2.set_label("Diff Magnitude (deg)", fontsize=9)
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
    plot_stats_text(ax1, cam1_x_position_diff_mean, cam1_x_position_diff_std)
    ax2 = axes[0, 2]
    ax2.plot(
        frame_cam1,
        gdf_cam1.y_position_diff,
        c="#4169E1",
        lw=1,
        label="Y position (northing)",
    )
    plot_stats_text(ax2, cam1_y_position_diff_mean, cam1_y_position_diff_std)
    ax3 = axes[0, 3]
    ax3.plot(
        frame_cam1,
        gdf_cam1.z_position_diff,
        c="#87CEEB",
        lw=1,
        label="Z position (altitude)",
    )
    plot_stats_text(ax3, cam1_z_position_diff_mean, cam1_z_position_diff_std)

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
        ax.hlines(
            0, frame_cam1.min(), frame_cam1.max(), color="k", linestyle="-", lw=0.5
        )
        ax.set_title("Camera 1", loc="right", fontsize=10, y=0.98)
        ax.set_xlabel("Linescan Sample", fontsize=9)
        ax.set_ylabel("Original $-$ Optimized (m)", fontsize=9)
        if shared_scales:
            ax.set_ylim(min_val_position_diff, max_val_position_diff)
        if log_scale_positions:
            ax.set_yscale("symlog")
        ax.set_xlim(frame_cam1.min(), frame_cam1.max())
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=9)

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
    plot_stats_text(ax1_r, cam1_roll_diff_mean, cam1_roll_diff_std)

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
    plot_stats_text(ax2_r, cam1_pitch_diff_mean, cam1_pitch_diff_std)

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
    plot_stats_text(ax3_r, cam1_yaw_diff_mean, cam1_yaw_diff_std)

    # Share y-axis for angular diff plots
    min_val_angle_diff = min(
        gdf_cam1.roll_diff.min(), gdf_cam1.pitch_diff.min(), gdf_cam1.yaw_diff.min()
    )
    max_val_angle_diff = max(
        gdf_cam1.roll_diff.max(), gdf_cam1.pitch_diff.max(), gdf_cam1.yaw_diff.max()
    )

    for ax, ax_r in [(ax1, ax1_r), (ax2, ax2_r), (ax3, ax3_r)]:
        ax.hlines(
            0, frame_cam1.min(), frame_cam1.max(), color="k", linestyle="-", lw=0.5
        )
        ax.set_title("Camera 1", loc="right", fontsize=10, y=0.98)
        ax.set_xlabel("Linescan Sample", fontsize=9)
        ax.set_ylabel("Original $-$ Optimized (deg)", fontsize=9)
        if shared_scales:
            ax.set_ylim(min_val_angle_diff, max_val_angle_diff)
        ax_r.set_ylabel("Original (deg)", fontsize=9)
        if log_scale_angles:
            ax.set_yscale("symlog")
        ax.set_xlim(frame_cam1.min(), frame_cam1.max())
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax_r.tick_params(axis="both", which="major", labelsize=9)
        lines1, labels1 = ax_r.get_legend_handles_labels()
        lines2, labels2 = ax.get_legend_handles_labels()
        ax_r.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Camera 2 mapview plot
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
        norm=Normalize(vmin=cam1_position_vmin, vmax=cam1_position_vmax), cmap="viridis"
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
        norm=Normalize(vmin=cam1_angular_vmin, vmax=cam1_angular_vmax), cmap="inferno"
    )
    cbar2 = plt.colorbar(
        sm2, ax=ax, extend=extend, orientation="vertical", aspect=30, pad=0.05
    )
    cbar2.set_label("Diff Magnitude (deg)", fontsize=9)
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
    plot_stats_text(ax1, cam2_x_position_diff_mean, cam2_x_position_diff_std)
    ax2 = axes[2, 2]
    ax2.plot(
        frame_cam2,
        gdf_cam2.y_position_diff,
        c="#4169E1",
        lw=1,
        label="Y position (northing)",
    )
    plot_stats_text(ax2, cam2_y_position_diff_mean, cam2_y_position_diff_std)
    ax3 = axes[2, 3]
    ax3.plot(
        frame_cam2,
        gdf_cam2.z_position_diff,
        c="#87CEEB",
        lw=1,
        label="Z position (altitude)",
    )
    plot_stats_text(ax3, cam2_z_position_diff_mean, cam2_z_position_diff_std)

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
        ax.hlines(
            0, frame_cam2.min(), frame_cam2.max(), color="k", linestyle="-", lw=0.5
        )
        ax.set_title("Camera 2", loc="right", fontsize=10, y=0.98)
        ax.set_xlabel("Linescan Sample", fontsize=9)
        ax.set_ylabel("Original $-$ Optimized (m)", fontsize=9)
        if shared_scales:
            ax.set_ylim(min_val_position_diff, max_val_position_diff)
        if log_scale_positions:
            ax.set_yscale("symlog")
        ax.set_xlim(frame_cam2.min(), frame_cam2.max())
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=9)

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
    plot_stats_text(ax1_r, cam2_roll_diff_mean, cam2_roll_diff_std)

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
    plot_stats_text(ax2_r, cam2_pitch_diff_mean, cam2_pitch_diff_std)

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
    plot_stats_text(ax3_r, cam2_yaw_diff_mean, cam2_yaw_diff_std)

    # Share y-axis for angular diff plots
    min_val_angle_diff = min(
        gdf_cam2.roll_diff.min(), gdf_cam2.pitch_diff.min(), gdf_cam2.yaw_diff.min()
    )
    max_val_angle_diff = max(
        gdf_cam2.roll_diff.max(), gdf_cam2.pitch_diff.max(), gdf_cam2.yaw_diff.max()
    )

    for ax, ax_r in [(ax1, ax1_r), (ax2, ax2_r), (ax3, ax3_r)]:
        ax.hlines(
            0, frame_cam2.min(), frame_cam2.max(), color="k", linestyle="-", lw=0.5
        )
        ax.set_title("Camera 2", loc="right", fontsize=10, y=0.98)
        ax.set_xlabel("Linescan Sample", fontsize=9)
        ax.set_ylabel("Original $-$ Optimized (deg)", fontsize=9)
        if shared_scales:
            ax.set_ylim(min_val_angle_diff, max_val_angle_diff)
        ax_r.set_ylabel("Original (deg)", fontsize=9)
        if log_scale_angles:
            ax.set_yscale("symlog")
        ax.set_xlim(frame_cam2.min(), frame_cam2.max())
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax_r.tick_params(axis="both", which="major", labelsize=9)
        lines1, labels1 = ax_r.get_legend_handles_labels()
        lines2, labels2 = ax.get_legend_handles_labels()
        ax_r.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Set linewidth and color for all spines
    for ax in axes[:2].flatten():
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("#141414")

    for ax in axes[2:].flatten():
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color("#A9A9A9")

    if title:
        title_text = f"{title}: Position and Angle Changes for Camera 1 ({cam1_name}) and Camera 2 ({cam2_name})"
    else:
        title_text = f"Position and Angle Changes for Camera 1 ({cam1_name}) and Camera 2 ({cam2_name})"
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
# Methods below copied from orbit_plot.py in the ASP source code on 15-Aug-2024:
# https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/orbit_plot.py
#


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
