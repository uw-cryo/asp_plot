import geopandas as gpd
import matplotlib
import pandas as pd
import pytest

from asp_plot.bundle_adjust import PlotBundleAdjustFiles, ReadBundleAdjustFiles

matplotlib.use("Agg")


class TestBundleAdjust:
    @pytest.fixture
    def ba_files(self):
        directory = "tests/test_data"
        ba_directory = "ba"
        return ReadBundleAdjustFiles(directory, ba_directory)

    @pytest.fixture
    def ba_files_no_mapproj_dem(self):
        directory = "tests/test_data"
        ba_directory = "ba_no_mapproj_dem"
        return ReadBundleAdjustFiles(directory, ba_directory)

    def test_get_initial_final_residuals_gdfs(self, ba_files):
        resid_initial, resid_final = ba_files.get_initial_final_residuals_gdfs()
        assert isinstance(resid_initial, gpd.GeoDataFrame)
        assert isinstance(resid_final, gpd.GeoDataFrame)

    def test_get_initial_final_geodiff_gdfs(self, ba_files):
        geodiff_initial, geodiff_final = ba_files.get_initial_final_residuals_gdfs(
            residuals_in_meters=True
        )
        assert isinstance(geodiff_initial, gpd.GeoDataFrame)
        assert isinstance(geodiff_final, gpd.GeoDataFrame)

    def test_get_mapproj_residuals_gdf(self, ba_files):
        resid_mapprojected_gdf = ba_files.get_mapproj_residuals_gdf()
        assert isinstance(resid_mapprojected_gdf, gpd.GeoDataFrame)

    def test_get_propagated_triangulation_uncert_df(self, ba_files):
        resid_triangulation_uncert_df = (
            ba_files.get_propagated_triangulation_uncert_df()
        )
        assert isinstance(resid_triangulation_uncert_df, pd.DataFrame)

    def test_plot_n_gdfs(self, ba_files):
        resid_initial, resid_final = ba_files.get_initial_final_residuals_gdfs()
        try:
            PlotBundleAdjustFiles([resid_initial, resid_final]).plot_n_gdfs(
                column_name="mean_residual"
            )
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_get_initial_final_geodiff_gdfs_no_mapproj_dem(
        self, ba_files_no_mapproj_dem
    ):
        """Test that geodiff gracefully fails when no --mapproj-dem was used in bundle_adjust."""
        with pytest.raises(ValueError) as exc_info:
            ba_files_no_mapproj_dem.get_initial_final_geodiff_gdfs()
        assert "could not be generated" in str(exc_info.value)

    def test_residuals_still_work_without_mapproj_dem(self, ba_files_no_mapproj_dem):
        """Test that residual GeoDataFrames can still be read even without --mapproj-dem."""
        resid_initial, resid_final = (
            ba_files_no_mapproj_dem.get_initial_final_residuals_gdfs()
        )
        assert isinstance(resid_initial, gpd.GeoDataFrame)
        assert isinstance(resid_final, gpd.GeoDataFrame)
