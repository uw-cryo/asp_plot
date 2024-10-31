import pytest

from asp_plot.processing_parameters import ProcessingParameters


class TestProcessingParameters:
    @pytest.fixture
    def processing_parameters(self):
        processing_parameters = ProcessingParameters(
            processing_directory="tests/test_data",
            bundle_adjust_directory="ba",
            stereo_directory="stereo",
        )
        return processing_parameters

    @pytest.fixture
    def processing_parameters_no_ba(self):
        processing_parameters = ProcessingParameters(
            processing_directory="tests/test_data",
            stereo_directory="stereo",
        )
        return processing_parameters

    def test_init(self, processing_parameters):
        assert processing_parameters.bundle_adjust_log is not None
        assert processing_parameters.stereo_logs is not None
        assert processing_parameters.point2dem_log is not None

    def test_from_log_files(self, processing_parameters):
        result = processing_parameters.from_log_files()
        assert result["processing_timestamp"] != ""
        assert result["reference_dem"] != ""
        assert result["bundle_adjust"] != ""
        assert result["bundle_adjust_run_time"] != "N/A"
        assert result["stereo"] != ""
        assert result["stereo_run_time"] != "N/A"
        assert result["point2dem"] != ""
        assert result["point2dem_run_time"] != "N/A"

    def test_from_log_files_no_ba(self, processing_parameters_no_ba):
        result = processing_parameters_no_ba.from_log_files()
        assert result["bundle_adjust"] == "Bundle adjustment not run"
        assert result["bundle_adjust_run_time"] == "N/A"
