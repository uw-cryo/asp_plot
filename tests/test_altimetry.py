import geopandas as gpd
import matplotlib
import pytest

from asp_plot.altimetry import Altimetry

matplotlib.use("Agg")


class TestAltimetry:
    @pytest.fixture
    def icesat(self):
        icesat = Altimetry(
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )
        icesat.atl06sr_filtered = gpd.read_parquet(
            "tests/test_data/icesat_data/atl06sr_defaults_filtered.parquet"
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

    def test_plot_atl06sr(self, icesat):
        try:
            icesat.plot_atl06sr(filtered=True)
        except Exception as e:
            pytest.fail(f"plot_atl06sr() method raised an exception: {str(e)}")

    def test_atl06sr_to_dem_dh(self, icesat):
        try:
            icesat.atl06sr_to_dem_dh()
        except Exception as e:
            pytest.fail(f"atl06sr_to_dem_dh() method raised an exception: {str(e)}")

    def test_mapview_plot_atl06sr_to_dem(self, icesat):
        try:
            icesat.mapview_plot_atl06sr_to_dem()
        except Exception as e:
            pytest.fail(
                f"mapview_plot_atl06sr_to_dem() method raised an exception: {str(e)}"
            )

    def test_histogram(self, icesat):
        try:
            icesat.histogram()
        except Exception as e:
            pytest.fail(f"histogram() method raised an exception: {str(e)}")
