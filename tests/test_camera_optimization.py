import matplotlib
import pytest

from asp_plot.camera_optimization import summary_plot_two_camera_optimization

matplotlib.use("Agg")


class TestCameraOptimization:
    def test_summary_plot_two_camera_optimization(self):
        try:
            summary_plot_two_camera_optimization(
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
