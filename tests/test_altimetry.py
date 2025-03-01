import geopandas as gpd
import matplotlib
import pytest

from asp_plot.altimetry import Altimetry

matplotlib.use("Agg")


class TestAltimetry:
    @pytest.fixture
    def icesat(self):
        icesat = Altimetry(
            directory="tests/test_data",
            dem_fn="tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
        )

        # The actual request is fairly time consuming, thus better to be skipped for testing.
        # icesat.request_atl06sr_multi_processing(
        #     filename="tests/test_data/icesat_data/atl06sr_",
        # )
        for key in ["all", "ground", "canopy", "top_of_canopy"]:
            icesat.atl06sr_processing_levels[key] = gpd.read_parquet(
                f"tests/test_data/icesat_data/atl06sr_{key}.parquet"
            )
            icesat.atl06sr_processing_levels_filtered[key] = (
                icesat.atl06sr_processing_levels[key]
            )
        return icesat

    def test_filter_esa_worldcover(self, icesat):
        try:
            icesat.filter_esa_worldcover()
        except Exception as e:
            pytest.fail(f"filter_esa_worldcover() method raised an exception: {str(e)}")

    def test_generic_temporal_filter_atl06sr(self, icesat):
        try:
            icesat.generic_temporal_filter_atl06sr(
                select_years=[2019, 2020],
                select_months=[1, 2, 3],
                select_days=[1, 2, 3],
            )
        except Exception as e:
            pytest.fail(
                f"generic_temporal_filter_atl06sr() method raised an exception: {str(e)}"
            )

    def test_predefined_temporal_filter_atl06sr(self, icesat):
        try:
            icesat.predefined_temporal_filter_atl06sr()
        except Exception as e:
            pytest.fail(
                f"predefined_temporal_filter_atl06sr() method raised an exception: {str(e)}"
            )

    def test_plot_atl06sr_time_stamps(self, icesat):
        try:
            icesat.plot_atl06sr_time_stamps()
        except Exception as e:
            pytest.fail(
                f"plot_atl06sr_time_stamps() method raised an exception: {str(e)}"
            )

    def test_plot_atl06sr(self, icesat):
        try:
            icesat.plot_atl06sr()
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
            icesat.histogram(key="ground_45_day_pad")
        except Exception as e:
            pytest.fail(f"histogram() method raised an exception: {str(e)}")

    def test_alignment_report(self, icesat):
        try:
            icesat.alignment_report()
        except Exception as e:
            pytest.fail(f"alignment_report() method raised an exception: {str(e)}")
