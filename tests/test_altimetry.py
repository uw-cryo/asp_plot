import matplotlib
import pytest

from asp_plot.altimetry import Altimetry

matplotlib.use("Agg")


class TestAltimetry:
    @pytest.fixture
    def icesat(self):
        icesat = Altimetry(
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
            geojson_fn="tests/test_data/icesat_region.geojson",
        )
        return icesat

    def test_pull_and_filter_atl06sr(self, icesat):
        try:
            icesat.pull_atl06sr(esa_worldcover=False, save_to_parquet=False)
            icesat.filter_atl06sr(
                mask_worldcover_water=False, save_to_parquet=False, save_to_csv=False
            )
        except Exception as e:
            pytest.fail(
                f"pull_atl06sr() or filter_atl06sr() method raised an exception: {str(e)}"
            )
