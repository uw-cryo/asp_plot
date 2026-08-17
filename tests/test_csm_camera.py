import json
import os

import matplotlib
import numpy as np
import pytest

from asp_plot.csm_camera import (
    csm_camera_summary_plot,
    get_orbit_plot_gdf,
    read_angles,
    read_csm_cam,
    reproject_ecef,
    wrap_angle_diff,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

CAM1 = [
    "tests/test_data/jitter/uyuni/csm-104001001427B900.r100.adjusted_state.json",
    "tests/test_data/jitter/uyuni/jitter_solved_run-csm-104001001427B900.r100.adjusted_state.json",
]
CAM2 = [
    "tests/test_data/jitter/uyuni/csm-1040010014761800.r100.adjusted_state.json",
    "tests/test_data/jitter/uyuni/jitter_solved_run-csm-1040010014761800.r100.adjusted_state.json",
]

GOLDEN_FN = "tests/test_data/jitter/uyuni/csm_camera_golden.json"

# Columns whose summary statistics are pinned by the golden GDF characterization.
DIFF_COLS = [
    "position_diff_magnitude",
    "x_position_diff",
    "y_position_diff",
    "z_position_diff",
    "angular_diff_magnitude",
    "roll_diff",
    "pitch_diff",
    "yaw_diff",
    "original_roll",
    "original_pitch",
    "original_yaw",
]


def _load_golden():
    with open(GOLDEN_FN, "r") as f:
        return json.load(f)


def _gdf_summary(gdf):
    out = {
        "shape": list(gdf.shape),
        "columns": list(gdf.columns),
        "crs": str(gdf.crs),
        "has_line_at_position": "line_at_position" in gdf.columns,
        "stats": {},
    }
    for col in DIFF_COLS:
        s = gdf[col]
        out["stats"][col] = [
            round(float(s.mean()), 9),
            round(float(s.std()), 9),
            round(float(s.min()), 9),
            round(float(s.max()), 9),
        ]
    if out["has_line_at_position"]:
        lap = gdf["line_at_position"]
        out["line_at_position_first_last"] = [int(lap.iloc[0]), int(lap.iloc[-1])]
    return out


def _fig_capture(fig):
    """Capture the platform-stable structure of a figure.

    Only string/int fields (titles, axis labels, line counts, axes count,
    suptitle) are recorded -- these are reproducible across numpy/scipy/
    matplotlib versions. The numeric content of the plotted lines is verified
    separately, with tolerance, by ``_assert_camera_lines``.
    """
    axes = fig.axes
    cap = {
        "n_axes": len(axes),
        "suptitle": fig._suptitle.get_text() if fig._suptitle else None,
        "axes": [],
    }
    for ax in axes:
        cap["axes"].append(
            {
                "title": ax.get_title(),
                "title_right": ax.get_title(loc="right"),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "n_lines": len(ax.get_lines()),
            }
        )
    return cap


def _assert_camera_lines(main_axes, gdf):
    """Assert a camera's line panels plot that camera's GDF columns.

    ``main_axes`` is the camera's 8 grid axes in row-major order:
    ``[pos_map, x, y, z, ang_map, roll, pitch, yaw]``. This is what pins the
    cam1/cam2 collapse -- it proves each camera's panels are fed that camera's
    data (e.g. camera 2's x panel must carry ``gdf_cam2.x_position_diff``).
    Compared with tolerance so it is robust across platforms.
    """
    expected = {
        1: gdf.x_position_diff.values,
        2: gdf.y_position_diff.values,
        3: gdf.z_position_diff.values,
        5: gdf.roll_diff.values,
        6: gdf.pitch_diff.values,
        7: gdf.yaw_diff.values,
    }
    for idx, want in expected.items():
        got = np.asarray(main_axes[idx].get_lines()[0].get_ydata(), dtype=float)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def _write_resampled_camera(src_fn, dst_fn, factor=2, position_noise_m=0.0, seed=0):
    """Write a copy of a linescan CSM camera on a denser ephemeris/quaternion grid.

    This is what ``bundle_adjust`` and ``jitter_solve`` do to their input: they
    resample the trajectory to a finer spacing and nudge the positions. The
    *orientations* here are only resampled, never rotated, so the true
    orientation change between ``src_fn`` and the written camera is zero.
    """
    model = read_csm_cam(src_fn)

    def densify(flat, width):
        arr = np.asarray(flat, dtype=float).reshape(-1, width)
        n_out = (arr.shape[0] - 1) * factor + 1
        x_in = np.linspace(0, 1, arr.shape[0])
        x_out = np.linspace(0, 1, n_out)
        return np.column_stack(
            [np.interp(x_out, x_in, arr[:, i]) for i in range(width)]
        )

    positions = densify(model["m_positions"], 3)
    velocities = densify(model["m_velocities"], 3)
    quaternions = densify(model["m_quaternions"], 4)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    if position_noise_m:
        rng = np.random.default_rng(seed)
        # Perturb along the radial (nadir) direction, which is where solver
        # position changes mostly land, and where they most strongly tilt an
        # along-track direction estimated by central differences.
        radial = positions / np.linalg.norm(positions, axis=1, keepdims=True)
        positions = positions + radial * rng.normal(
            0.0, position_noise_m, size=(positions.shape[0], 1)
        )

    model["m_positions"] = positions.ravel().tolist()
    model["m_velocities"] = velocities.ravel().tolist()
    model["m_quaternions"] = quaternions.ravel().tolist()
    model["m_numPositions"] = positions.size
    model["m_numQuaternions"] = quaternions.size
    model["m_dtEphem"] = model["m_dtEphem"] / factor
    model["m_dtQuat"] = model["m_dtQuat"] / factor

    with open(dst_fn, "w") as f:
        f.write("USGS_ASTRO_LINE_SCANNER_SENSOR_MODEL\n")
        json.dump(model, f)
    return dst_fn


class TestAngleReferenceFrame:
    """Angle differences must measure the cameras, not the estimated frame.

    ``csm_io.read_angles()`` mirrors ASP's ``orbit_plot.py``, which estimates
    the satellite body frame separately for each camera from a central
    difference of that camera's own ephemeris. Differencing two such
    independently-estimated frames injects a large spurious signal, because a
    solver both resamples the ephemeris to a finer spacing (shortening the
    central-difference baseline) and perturbs the positions. See issue #53.
    """

    def test_diff_of_identical_cameras_is_zero(self):
        gdf = get_orbit_plot_gdf(CAM1[0], CAM1[0])
        for col in ["roll_diff", "pitch_diff", "yaw_diff"]:
            np.testing.assert_allclose(gdf[col].values, 0.0, atol=1e-12)

    def test_resampled_and_nudged_ephemeris_does_not_fake_a_rotation(self, tmp_path):
        # Same orientations, denser trajectory, 2 m of radial position noise:
        # the reported orientation change must stay near zero.
        optimized = _write_resampled_camera(
            CAM1[0], str(tmp_path / "resampled.json"), factor=2, position_noise_m=2.0
        )
        gdf = get_orbit_plot_gdf(CAM1[0], optimized)

        for col in ["roll_diff", "pitch_diff", "yaw_diff"]:
            assert (
                np.abs(gdf[col].values).max() < 1e-3
            ), f"{col} reports a rotation that the camera did not undergo"

        # Guard the regression the other way round: ASP's per-camera frames do
        # see this as a large rotation, so the assertion above is not vacuous.
        original_angles, optimized_angles = read_angles([CAM1[0]], [optimized], [])
        asp_pitch = np.array([a[1] for a in original_angles])
        asp_pitch_opt = np.array([a[1] for a in optimized_angles])
        asp_pitch_diff = (
            np.interp(
                np.linspace(0, 1, len(asp_pitch_opt)),
                np.linspace(0, 1, len(asp_pitch)),
                asp_pitch,
            )
            - asp_pitch_opt
        )
        assert np.abs(asp_pitch_diff).max() > 0.1


class TestAngleWrapping:
    def test_wrap_angle_diff(self):
        np.testing.assert_allclose(
            wrap_angle_diff([0.0, 10.0, -10.0, 359.8, -359.8, 180.0, -180.0]),
            [0.0, 10.0, -10.0, -0.2, 0.2, -180.0, -180.0],
            atol=1e-9,
        )

    def test_yaw_near_the_branch_cut_is_not_a_360_degree_change(self):
        # A camera at yaw ~+179.9 compared against one at ~-179.9 changed by
        # 0.2 degrees, not by 359.8.
        np.testing.assert_allclose(wrap_angle_diff(179.9 - -179.9), -0.2, atol=1e-9)


class TestCameraOptimization:
    def test_csm_camera_summary_plot(self):
        try:
            csm_camera_summary_plot(CAM1, CAM2)
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
        finally:
            plt.close("all")


class TestCsmCameraCharacterization:
    """Golden characterization pinned before the #131 structural split.

    These tests lock in the exact numeric output of ``get_orbit_plot_gdf`` and
    the full figure structure (per-axis titles/labels + a digest of every
    plotted line) of ``csm_camera_summary_plot``. The line digests are what
    guarantee the cam1/cam2 plotting halves stay behavior-identical after they
    are collapsed into a single helper.
    """

    def test_get_orbit_plot_gdf_golden(self):
        golden = _load_golden()["gdf_cam1"]
        gdf = get_orbit_plot_gdf(CAM1[0], CAM1[1])
        summary = _gdf_summary(gdf)

        assert summary["shape"] == golden["shape"]
        assert summary["columns"] == golden["columns"]
        assert summary["crs"] == golden["crs"]
        assert summary["has_line_at_position"] == golden["has_line_at_position"]
        assert (
            summary["line_at_position_first_last"]
            == golden["line_at_position_first_last"]
        )
        for col in DIFF_COLS:
            assert summary["stats"][col] == pytest.approx(
                golden["stats"][col], abs=1e-6
            ), f"GDF stats drifted for column {col}"

    def test_get_orbit_plot_gdf_map_crs_reprojects(self):
        # map_crs should reproject ECEF -> the requested EPSG without changing
        # the number of samples or the angular columns.
        gdf_ecef = get_orbit_plot_gdf(CAM1[0], CAM1[1])
        gdf_utm = get_orbit_plot_gdf(CAM1[0], CAM1[1], map_crs=32719)
        assert str(gdf_ecef.crs) == "EPSG:4978"
        assert str(gdf_utm.crs) == "EPSG:32719"
        assert gdf_ecef.shape == gdf_utm.shape
        # Angular diffs are CRS-independent; they must be untouched.
        np.testing.assert_allclose(gdf_ecef.roll_diff.values, gdf_utm.roll_diff.values)

    def test_reproject_ecef_roundtrip(self):
        # A point on the equator/prime-meridian-ish ECEF coordinate reprojects
        # to lon/lat that round-trips back within tolerance.
        positions = np.array([[6378137.0, 0.0, 0.0]])
        out = reproject_ecef(positions, to_epsg=4326)
        assert out.shape == (1, 3)
        lat, lon = out[0, 0], out[0, 1]
        assert lat == pytest.approx(0.0, abs=1e-6)
        assert lon == pytest.approx(0.0, abs=1e-6)

    def test_csm_camera_summary_plot_two_cam_golden(self):
        golden = _load_golden()["plot_two_cam"]
        gdf_cam1 = get_orbit_plot_gdf(CAM1[0], CAM1[1])
        gdf_cam2 = get_orbit_plot_gdf(CAM2[0], CAM2[1])
        plt.close("all")
        csm_camera_summary_plot(CAM1, CAM2)
        fig = plt.gcf()
        cap = _fig_capture(fig)
        # fig.axes[0:8] are camera 1's grid (rows 0-1), [8:16] camera 2's
        # (rows 2-3), both in row-major order; later axes are colorbars/twins.
        _assert_camera_lines(fig.axes[0:8], gdf_cam1)
        _assert_camera_lines(fig.axes[8:16], gdf_cam2)
        plt.close("all")
        assert cap == golden

    def test_csm_camera_summary_plot_one_cam_golden(self):
        golden = _load_golden()["plot_one_cam"]
        gdf_cam1 = get_orbit_plot_gdf(CAM1[0], CAM1[1])
        plt.close("all")
        csm_camera_summary_plot(CAM1)
        fig = plt.gcf()
        cap = _fig_capture(fig)
        _assert_camera_lines(fig.axes[0:8], gdf_cam1)
        plt.close("all")
        assert cap == golden


class TestCsmCameraSaveOutput:
    def test_csm_camera_summary_plot_saves_png(self, tmp_path):
        fig_fn = "csm_camera_summary_plot.png"
        csm_camera_summary_plot(CAM1, CAM2, save_dir=str(tmp_path), fig_fn=fig_fn)
        plt.close("all")
        assert os.path.exists(os.path.join(str(tmp_path), fig_fn))
