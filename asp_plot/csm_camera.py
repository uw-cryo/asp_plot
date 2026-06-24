"""
Plotting for CSM camera optimization / jitter results.

This module is the plotting layer of the former monolithic ``csm_camera.py``.
The numeric analysis now lives in :mod:`asp_plot.csm_analysis` and the
ASP-mirrored camera-model readers in :mod:`asp_plot.csm_io`. Those symbols are
re-exported here for backward compatibility with notebooks and downstream code
that imported them from ``asp_plot.csm_camera``.
"""

import os

import contextily as ctx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from asp_plot.csm_analysis import (  # noqa: F401  (re-exported for back-compat)
    get_orbit_plot_gdf,
    poly_fit,
    reproject_ecef,
)
from asp_plot.csm_io import (  # noqa: F401  (re-exported for back-compat)
    ASP_TO_CSM_SHIFT,
    estim_satellite_orientation,
    getLineAtTime,
    getTimeAtLine,
    isLinescan,
    read_angles,
    read_csm_cam,
    read_frame_cam_dict,
    read_frame_csm_cam,
    read_linescan_pos_rot,
    read_positions_rotations,
    read_positions_rotations_from_file,
    read_tsai_cam,
    roll_pitch_yaw,
    toCsmPixel,
    tsai_list_to_gdf,
)
from asp_plot.utils import save_figure


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


def _apply_frame_xaxis(ax, gdf, frame):
    """
    Label the x-axis of a per-sample line plot.

    For linescan cameras (where ``gdf`` carries a ``line_at_position`` column)
    the ticks are relabeled with the corresponding image line numbers; otherwise
    the axis is simply labeled by position sample. Shared by the position and
    angle line panels.
    """
    if "line_at_position" in gdf.columns:
        ax.set_xlabel("Linescan Image Line Number", fontsize=9)
        xticks = ax.get_xticks()
        xticks = xticks[(xticks >= frame.min()) & (xticks <= frame.max())]
        xtick_labels = [
            (
                gdf.iloc[int(x)].line_at_position
                if int(x) < len(gdf.line_at_position)
                else ""
            )
            for x in xticks
        ]
        xtick_labels = [np.round(x, -1) for x in xtick_labels]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
    else:
        ax.set_xlabel("Position Sample", fontsize=9)


def _plot_camera(
    pos_row,
    ang_row,
    gdf,
    cam_name,
    cam_label,
    position_vmin,
    position_vmax,
    angular_vmin,
    angular_vmax,
    extend,
    add_basemap,
    shared_scales,
    log_scale_positions,
    log_scale_angles,
    ctx_kwargs,
):
    """
    Render the full set of panels for a single camera.

    Draws, in upstream creation order, the position-magnitude map, the
    angle-magnitude map, the x/y/z position-difference line panels, and the
    roll/pitch/yaw angle-difference line panels (with twin axes for the original
    angles). Called once per camera by :func:`csm_camera_summary_plot`; this is
    the collapse of the formerly near-verbatim cam1/cam2 halves.

    Parameters
    ----------
    pos_row, ang_row : sequence of matplotlib.axes.Axes
        The length-4 axis rows for this camera's position row and angle row.
    gdf : geopandas.GeoDataFrame
        Output of :func:`asp_plot.csm_analysis.get_orbit_plot_gdf`.
    cam_name : str
        Short camera name shown in the map-panel titles.
    cam_label : str
        Row label, e.g. ``"Camera 1"``.
    position_vmin, position_vmax, angular_vmin, angular_vmax : float
        Colorbar limits (shared across cameras; computed from camera 1).
    extend : str
        Colorbar ``extend`` mode.
    add_basemap : bool
        Whether to add a contextily basemap to the map panels.
    shared_scales, log_scale_positions, log_scale_angles : bool
        Line-panel y-axis options.
    ctx_kwargs : dict
        Extra keyword arguments forwarded to ``contextily.add_basemap``.
    """
    # Per-camera statistics for the annotation boxes
    x_mean, x_std = gdf.x_position_diff.mean(), gdf.x_position_diff.std()
    y_mean, y_std = gdf.y_position_diff.mean(), gdf.y_position_diff.std()
    z_mean, z_std = gdf.z_position_diff.mean(), gdf.z_position_diff.std()
    roll_mean, roll_std = gdf.roll_diff.mean(), gdf.roll_diff.std()
    pitch_mean, pitch_std = gdf.pitch_diff.mean(), gdf.pitch_diff.std()
    yaw_mean, yaw_std = gdf.yaw_diff.mean(), gdf.yaw_diff.std()

    # Position-magnitude mapview plot
    ax = pos_row[0]
    gdf.plot(
        ax=ax,
        column="position_diff_magnitude",
        cmap="viridis",
        markersize=10,
        vmin=position_vmin,
        vmax=position_vmax,
    )
    ax.set_title(f"{cam_label}: Position Change\n{cam_name}", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    if add_basemap:
        ctx.add_basemap(ax=ax, **ctx_kwargs)
    sm1 = ScalarMappable(
        norm=Normalize(vmin=position_vmin, vmax=position_vmax), cmap="viridis"
    )
    cbar1 = plt.colorbar(
        sm1, ax=ax, extend=extend, orientation="vertical", aspect=30, pad=0.05
    )
    cbar1.set_label("Diff Magnitude (m)", fontsize=9)
    cbar1.ax.tick_params(labelsize=9)

    # Angle-magnitude mapview plot
    ax = ang_row[0]
    gdf.plot(
        ax=ax,
        column="angular_diff_magnitude",
        cmap="inferno",
        markersize=10,
        vmin=angular_vmin,
        vmax=angular_vmax,
    )
    ax.set_title(f"{cam_label}: Angle Change\n{cam_name}", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Easting (m)", fontsize=9)
    ax.set_ylabel("Northing (m)", fontsize=9)
    if add_basemap:
        ctx.add_basemap(ax=ax, **ctx_kwargs)
    sm2 = ScalarMappable(
        norm=Normalize(vmin=angular_vmin, vmax=angular_vmax), cmap="inferno"
    )
    cbar2 = plt.colorbar(
        sm2, ax=ax, extend=extend, orientation="vertical", aspect=30, pad=0.05
    )
    cbar2.set_label("Diff Magnitude (°)", fontsize=9)
    cbar2.ax.tick_params(labelsize=9)

    frame = np.arange(gdf.shape[0])

    # Position difference (x, y, z) line plots
    ax1 = pos_row[1]
    ax1.plot(
        frame, gdf.x_position_diff, c="#000080", lw=1, label="X position (easting)"
    )
    plot_stats_text(ax1, x_mean, x_std, unit="m")
    ax2 = pos_row[2]
    ax2.plot(
        frame, gdf.y_position_diff, c="#4169E1", lw=1, label="Y position (northing)"
    )
    plot_stats_text(ax2, y_mean, y_std, unit="m")
    ax3 = pos_row[3]
    ax3.plot(
        frame, gdf.z_position_diff, c="#87CEEB", lw=1, label="Z position (altitude)"
    )
    plot_stats_text(ax3, z_mean, z_std, unit="m")

    # Share y-axis for position diff plots
    min_val_position_diff = min(
        gdf.x_position_diff.min(),
        gdf.y_position_diff.min(),
        gdf.z_position_diff.min(),
    )
    max_val_position_diff = max(
        gdf.x_position_diff.max(),
        gdf.y_position_diff.max(),
        gdf.z_position_diff.max(),
    )

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(frame.min(), frame.max())
        ax.hlines(0, frame.min(), frame.max(), color="k", linestyle="-", lw=0.5)
        ax.set_title(cam_label, loc="right", fontsize=10, y=0.98)
        _apply_frame_xaxis(ax, gdf, frame)
        ax.set_ylabel("Original $-$ Optimized (m)", fontsize=9)
        if shared_scales:
            ax.set_ylim(min_val_position_diff, max_val_position_diff)
        if log_scale_positions:
            ax.set_yscale("symlog")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.8, color="gray")
        ax.legend(loc="upper right", fontsize=8)
        ax.tick_params(axis="both", which="major", labelsize=8)

    # Angle difference (roll, pitch, yaw) line plots
    ax1 = ang_row[1]
    ax1.plot(frame, gdf.roll_diff, c="#FF4500", lw=1, label="Roll Diff")
    ax1_r = ax1.twinx()
    ax1_r.plot(
        frame, gdf.original_roll, c="k", lw=1, linestyle="--", label="Original Roll"
    )
    plot_stats_text(ax1_r, roll_mean, roll_std, unit="°")

    ax2 = ang_row[2]
    ax2.plot(frame, gdf.pitch_diff, c="#FFA500", lw=1, label="Pitch Diff")
    ax2_r = ax2.twinx()
    ax2_r.plot(
        frame, gdf.original_pitch, c="k", lw=1, linestyle="--", label="Original Pitch"
    )
    plot_stats_text(ax2_r, pitch_mean, pitch_std, unit="°")

    ax3 = ang_row[3]
    ax3.plot(frame, gdf.yaw_diff, c="#FFB347", lw=1, label="Yaw Diff")
    ax3_r = ax3.twinx()
    ax3_r.plot(
        frame, gdf.original_yaw, c="k", lw=1, linestyle="--", label="Original Yaw"
    )
    plot_stats_text(ax3_r, yaw_mean, yaw_std, unit="°")

    # Share y-axis for angular diff plots
    min_val_angle_diff = min(
        gdf.roll_diff.min(), gdf.pitch_diff.min(), gdf.yaw_diff.min()
    )
    max_val_angle_diff = max(
        gdf.roll_diff.max(), gdf.pitch_diff.max(), gdf.yaw_diff.max()
    )

    for ax, ax_r in [(ax1, ax1_r), (ax2, ax2_r), (ax3, ax3_r)]:
        ax.set_xlim(frame.min(), frame.max())
        ax.hlines(0, frame.min(), frame.max(), color="k", linestyle="-", lw=0.5)
        ax.set_title(cam_label, loc="right", fontsize=10, y=0.98)
        _apply_frame_xaxis(ax, gdf, frame)
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

    # Calculate colorbar ranges from camera 1; camera 2 reuses the same scale.
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

    # Begin plot
    if cam2_list:
        fig, axes = plt.subplots(4, 4, figsize=figsize)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(figsize[0], figsize[1] / 2))

    _plot_camera(
        axes[0],
        axes[1],
        gdf_cam1,
        cam1_name,
        "Camera 1",
        cam1_position_vmin,
        cam1_position_vmax,
        cam1_angular_vmin,
        cam1_angular_vmax,
        extend,
        add_basemap,
        shared_scales,
        log_scale_positions,
        log_scale_angles,
        ctx_kwargs,
    )

    if cam2_list:
        _plot_camera(
            axes[2],
            axes[3],
            gdf_cam2,
            cam2_name,
            "Camera 2",
            cam1_position_vmin,
            cam1_position_vmax,
            cam1_angular_vmin,
            cam1_angular_vmax,
            extend,
            add_basemap,
            shared_scales,
            log_scale_positions,
            log_scale_angles,
            ctx_kwargs,
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
