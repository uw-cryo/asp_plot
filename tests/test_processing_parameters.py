import pytest
from asp_plot.processing_parameters import ProcessingParameters
import matplotlib

matplotlib.use("Agg")


class TestProcessingParameters:
    @pytest.fixture
    def processing_parameters(self):
        processing_parameters = ProcessingParameters(
            directory="tests/test_data",
            bundle_adjust_directory="ba",
            stereo_directory="stereo",
        )
        return processing_parameters

    def test_init(self, processing_parameters):
        assert processing_parameters.bundle_adjust_log is not None
        assert processing_parameters.stereo_log is not None
        assert processing_parameters.point2dem_log is not None

    def test_from_log_files(self, processing_parameters):
        result = processing_parameters.from_log_files()
        assert "bundle_adjust" in result
        assert "stereo" in result
        assert "point2dem" in result
        assert "processing_timestamp" in result

    def test_plot_processing_parameters(self, processing_parameters):
        processing_parameters.from_log_files()
        try:
            processing_parameters.plot_processing_parameters()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
