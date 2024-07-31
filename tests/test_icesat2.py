import matplotlib
import pytest

from asp_plot.icesat2 import ICESat2

matplotlib.use("Agg")


class TestICESat2:
    @pytest.fixture
    def icesat(self):
        icesat = ICESat2(
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
            geojson_fn="tests/test_data/icesat_region.geojson",
        )
        return icesat

    def test_pull_and_clean_atl06_data(self, icesat):
        try:
            icesat.pull_atl06_data(esa_worldcover=False)
            icesat.clean_atl06(mask_worldcover_water=False)
        except Exception as e:
            pytest.fail(
                f"pull_atl06_data() or clean_atl06() method raised an exception: {str(e)}"
            )
