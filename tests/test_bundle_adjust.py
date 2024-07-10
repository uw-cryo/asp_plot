import pytest
from asp_plot.bundle_adjust import ReadBundleAdjustFiles, PlotBundleAdjustFiles
import matplotlib
import geopandas as gpd
import pandas as pd

matplotlib.use("Agg")


class TestBundleAdjust:
    @pytest.fixture
    def residual_files(self):
        directory = "tests/test_data"
        ba_directory = "ba"
        return ReadBundleAdjustFiles(directory, ba_directory)

    def test_get_initial_final_residuals_gdfs(self, residual_files):
        resid_initial, resid_final = residual_files.get_initial_final_residuals_gdfs()
        assert isinstance(resid_initial, gpd.GeoDataFrame)
        assert isinstance(resid_final, gpd.GeoDataFrame)

    def test_get_mapproj_residuals_gdf(self, residual_files):
        resid_mapprojected_gdf = residual_files.get_mapproj_residuals_gdf()
        assert isinstance(resid_mapprojected_gdf, gpd.GeoDataFrame)

    def test_get_propagated_triangulation_uncert_df(self, residual_files):
        resid_triangulation_uncert_df = residual_files.get_propagated_triangulation_uncert_df()
        assert isinstance(resid_triangulation_uncert_df, pd.DataFrame)

    def test_plot_n_gdfs(self, residual_files):
        resid_initial, resid_final = residual_files.get_initial_final_residuals_gdfs()
        try:
            PlotBundleAdjustFiles([resid_initial, resid_final]).plot_n_gdfs(column_name="mean_residual")
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
