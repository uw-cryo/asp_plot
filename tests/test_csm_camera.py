import matplotlib
import pytest

from asp_plot.csm_camera import csm_camera_summary_plot

matplotlib.use("Agg")


class TestCameraOptimization:
    def test_csm_camera_summary_plot(self):
        try:
            csm_camera_summary_plot(
                [
                    "tests/test_data/jitter/uyuni/csm-104001001427B900.r100.adjusted_state.json",
                    "tests/test_data/jitter/uyuni/jitter_solved_run-csm-104001001427B900.r100.adjusted_state.json",
                ],
                [
                    "tests/test_data/jitter/uyuni/csm-1040010014761800.r100.adjusted_state.json",
                    "tests/test_data/jitter/uyuni/jitter_solved_run-csm-1040010014761800.r100.adjusted_state.json",
                ],
            )
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
