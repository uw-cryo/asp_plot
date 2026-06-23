import hashlib
import json
import os

import matplotlib
import numpy as np
import pytest

from asp_plot.csm_camera import (
    csm_camera_summary_plot,
    get_orbit_plot_gdf,
    reproject_ecef,
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


def _line_digest(ax):
    """Stable digest of all Line2D y-data on an axis (rounded)."""
    h = hashlib.sha256()
    for ln in ax.get_lines():
        y = np.asarray(ln.get_ydata(), dtype=float)
        y = np.round(y, 6)
        h.update(y.tobytes())
    return h.hexdigest()[:16]


def _fig_capture(fig):
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
                "line_digest": _line_digest(ax),
            }
        )
    return cap


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
        plt.close("all")
        csm_camera_summary_plot(CAM1, CAM2)
        cap = _fig_capture(plt.gcf())
        plt.close("all")
        assert cap == golden

    def test_csm_camera_summary_plot_one_cam_golden(self):
        golden = _load_golden()["plot_one_cam"]
        plt.close("all")
        csm_camera_summary_plot(CAM1)
        cap = _fig_capture(plt.gcf())
        plt.close("all")
        assert cap == golden


class TestCsmCameraSaveOutput:
    def test_csm_camera_summary_plot_saves_png(self, tmp_path):
        fig_fn = "csm_camera_summary_plot.png"
        csm_camera_summary_plot(CAM1, CAM2, save_dir=str(tmp_path), fig_fn=fig_fn)
        plt.close("all")
        assert os.path.exists(os.path.join(str(tmp_path), fig_fn))
