import pytest
from asp_plot.bundle_adjust import ReadResiduals, PlotResiduals
import matplotlib
import geopandas as gpd
import pandas as pd

matplotlib.use("Agg")


class TestBundleAdjust:
    @pytest.fixture
    def residual_files(self):
        directory = "tests/test_data"
        ba_directory = "ba"
        return ReadResiduals(directory, ba_directory)

    def test_get_init_final_residuals_gdfs(self, residual_files):
        resid_init, resid_final = residual_files.get_init_final_residuals_gdfs()
        assert isinstance(resid_init, gpd.GeoDataFrame)
        assert isinstance(resid_final, gpd.GeoDataFrame)

    def test_get_mapproj_residuals_gdf(self, residual_files):
        resid_mapprojected_gdf = residual_files.get_mapproj_residuals_gdf()
        assert isinstance(resid_mapprojected_gdf, gpd.GeoDataFrame)

    def test_get_propagated_triangulation_uncert_df(self, residual_files):
        resid_triangulation_uncert_df = residual_files.get_propagated_triangulation_uncert_df()
        assert isinstance(resid_triangulation_uncert_df, pd.DataFrame)

    def test_plot_n_residuals(self, residual_files):
        resid_init, resid_final = residual_files.get_init_final_residuals_gdfs()
        try:
            PlotResiduals([resid_init, resid_final]).plot_n_residuals(column_name="mean_residual")
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
