import pytest

from asp_plot.processing_parameters import ProcessingParameters

# Golden values pinned from the real ASP 3.4.0-alpha log fixtures under
# tests/test_data/{ba,stereo}/. These lock in the exact parsed output before
# the #132 refactor moves log parsing behind a versioned adapter, so any
# behavioral drift in command/timestamp/DEM extraction is caught.
GOLDEN_PARAMETERS = {
    "asp_version": "3.4.0-alpha",
    "processing_timestamp": "2024-04-14 17:55:43",
    "reference_dem": "/nobackup/bpurint1/data/utqiagvik/COP/COP30_utqiagvik_lzw-adj_proj.tif",
    "bundle_adjust": (
        "bundle_adjust -t dg --weight-image /nobackup/bpurint1/data/utqiagvik/WV/"
        "utqiagvik_wv_EE/2022/utqiagvik_10m_UTM4N_seaice_mask_0and1.tif --datum WGS84 "
        "--individually-normalize --normalize-ip-tiles --ip-per-tile 50 "
        "--matches-per-tile 10 --min-triangulation-angle 10 --mapproj-dem "
        "/nobackup/bpurint1/data/utqiagvik/COP/COP30_utqiagvik_lzw-adj_proj.tif "
        "--propagate-errors --tri-weight 0.1 --tri-robust-threshold 0.1 "
        "--camera-weight 0 10300100D12D7400.r100.tif 10300100D0772D00.r100.tif "
        "10300100D12D7400.r100.xml 10300100D0772D00.r100.xml -o "
        "ba/ba_50ips_10matches_dg_weight_image --threads 28"
    ),
    "bundle_adjust_run_time": "0 hours and 3 minutes",
    "stereo": (
        "stereo --stereo-algorithm asp_mgm --corr-kernel 7 7 --subpixel-kernel 15 15 "
        "--cost-mode 4 --subpixel-mode 9 --corr-max-levels 5 --filter-mode 1 "
        "--erode-max-size 0 --individually-normalize --corr-memory-limit-mb 5000 "
        "--sgm-collar-size 256 --corr-tile-size 1024 --alignment-method none "
        "--corr-seed-mode 1 --compute-point-cloud-center-only --threads 24 "
        "1040010074793300_ortho_0.35m.tif 1040010075633C00_ortho_0.35m.tif "
        "ba/ba_50ips_10matches_dg_weight_image-1040010074793300.r100.adjusted_state.json "
        "ba/ba_50ips_10matches_dg_weight_image-1040010075633C00.r100.adjusted_state.json "
        "stereo/20220417_2252_1040010074793300_1040010075633C00 "
        "/nobackup/bpurint1/data/utqiagvik/COP/COP30_utqiagvik_lzw-adj_proj.tif"
    ),
    "stereo_run_time": "3 hours and 41 minutes",
    "point2dem": (
        "point2dem --nodata-value -9999 --t_srs EPSG:32604 --threads 24 "
        "--propagate-errors --remove-outliers --remove-outliers-params 75.0 3.0 "
        "--errorimage --tr 1 -o stereo_ba_50ips_10matches_dg_weight_image__ortho_0.55m"
        "_mode_asp_mgm_spm_9_corr_7_rfne_15_cost_4_refdem_COP30/"
        "20220419_2321_10300100D12D7400_10300100D0772D00_1m "
        "stereo_ba_50ips_10matches_dg_weight_image__ortho_0.55m_mode_asp_mgm_spm_9_"
        "corr_7_rfne_15_cost_4_refdem_COP30/"
        "20220419_2321_10300100D12D7400_10300100D0772D00-PC.tif"
    ),
    "point2dem_run_time": "0 hours and 14 minutes",
    # The stereo fixtures are not mapprojected scenes (no ASP mapproject GeoTIFF
    # metadata), so no commands are reconstructed.
    "mapproject": [],
}


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

    def test_asp_version(self, processing_parameters):
        version = processing_parameters.get_asp_version()
        assert version == "3.4.0-alpha"

    def test_asp_version_in_from_log_files(self, processing_parameters):
        result = processing_parameters.from_log_files()
        assert "asp_version" in result
        assert result["asp_version"] == "3.4.0-alpha"

    def test_from_log_files_no_ba(self, processing_parameters_no_ba):
        result = processing_parameters_no_ba.from_log_files()
        assert result["bundle_adjust"] == "Bundle adjustment not run"
        assert result["bundle_adjust_run_time"] == "N/A"

    def test_from_log_files_golden(self, processing_parameters):
        # Characterization: pin the exact parsed output of every field against
        # the real ASP 3.4.0-alpha log fixtures.
        result = processing_parameters.from_log_files()
        assert set(result.keys()) == set(GOLDEN_PARAMETERS.keys())
        for key, expected in GOLDEN_PARAMETERS.items():
            actual = result[key]
            if isinstance(expected, list):
                assert actual == expected, f"drift in field {key!r}"
            else:
                assert str(actual) == expected, f"drift in field {key!r}"
