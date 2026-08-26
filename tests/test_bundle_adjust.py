import os

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


class TestStemNarrowing:
    """An ASP-style prefix stem narrows the globs to that run's outputs (#60)."""

    def test_matching_stem_finds_files(self):
        ba_files = ReadBundleAdjustFiles("tests/test_data", "ba", stem="ba")
        initial, final = ba_files.get_csv_paths()
        assert initial.endswith("ba-initial_residuals_pointmap.csv")
        assert final.endswith("ba-final_residuals_pointmap.csv")

    def test_non_matching_stem_finds_nothing(self):
        ba_files = ReadBundleAdjustFiles("tests/test_data", "ba", stem="other_run")
        with pytest.raises(ValueError, match="not found"):
            ba_files.get_csv_paths()


class TestTwoStageDirectory:
    """A second bundle_adjust run (the pc_align two-stage workflow) leaves two
    generations of residual CSVs in one directory. The initial/final pair must
    come from the same generation, whatever order the filesystem lists them in
    (#6)."""

    @pytest.fixture
    def two_stage_directory(self, tmp_path):
        (tmp_path / "ba").mkdir()
        for stem in ("ba", "ba_pc_align"):
            for kind in ("initial", "final"):
                (tmp_path / "ba" / f"{stem}-{kind}_residuals_pointmap.csv").write_text(
                    ""
                )
        return tmp_path

    def test_initial_and_final_come_from_the_same_run(self, two_stage_directory):
        initial, final = ReadBundleAdjustFiles(
            str(two_stage_directory), "ba"
        ).get_csv_paths()
        assert os.path.basename(initial) == "ba-initial_residuals_pointmap.csv"
        assert os.path.basename(final) == "ba-final_residuals_pointmap.csv"

    def test_the_run_that_was_read_is_reported(self, two_stage_directory, caplog):
        with caplog.at_level("WARNING"):
            ReadBundleAdjustFiles(str(two_stage_directory), "ba").get_csv_paths()
        assert "ba_pc_align-final_residuals_pointmap.csv" in caplog.text
        assert "Using ba-final_residuals_pointmap.csv" in caplog.text

    def test_a_stem_selects_the_post_alignment_run(self, two_stage_directory, caplog):
        with caplog.at_level("WARNING"):
            initial, final = ReadBundleAdjustFiles(
                str(two_stage_directory), "ba", stem="ba_pc_align"
            ).get_csv_paths()
        assert os.path.basename(initial) == "ba_pc_align-initial_residuals_pointmap.csv"
        assert os.path.basename(final) == "ba_pc_align-final_residuals_pointmap.csv"
        assert caplog.text == ""
