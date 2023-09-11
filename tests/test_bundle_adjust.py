import pytest
from asp_plot.bundle_adjust import ReadResiduals, PlotResiduals
import matplotlib
import geopandas as gpd

matplotlib.use("Agg")


class TestBundleAdjust:
    @pytest.fixture
    def residual_files(self):
        directory = "tests/test_data/"
        ba_prefix = "ba/ba"
        return ReadResiduals(directory, ba_prefix)

    def test_get_residual_gdfs(self, residual_files):
        resid_init, resid_final = residual_files.get_residual_gdfs()
        assert isinstance(resid_init, gpd.GeoDataFrame)
        assert isinstance(resid_final, gpd.GeoDataFrame)

    def test_plot_n_residuals(self, residual_files):
        resid_init, resid_final = residual_files.get_residual_gdfs()
        try:
            PlotResiduals([resid_init, resid_final]).plot_n_residuals(column_name="mean_residual")
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
