import matplotlib
import pytest

from asp_plot.stereo import StereoFiles, StereoPlotter

matplotlib.use("Agg")


class TestStereoFiles:
    """File discovery is isolated in StereoFiles, separate from plotting."""

    @pytest.fixture
    def files(self):
        return StereoFiles(
            directory="tests/test_data",
            stereo_directory="stereo",
            dem_gsd=1,
            reference_dem="tests/test_data/ref_dem.tif",
        )

    def test_discovers_dem(self, files):
        assert files.dem_fn is not None
        assert files.dem_fn.endswith("-DEM.tif")

    def test_discovers_match_and_disparity(self, files):
        assert files.match_point_fn is not None
        assert "-disp-" not in files.match_point_fn

    def test_attribution_flag(self, files):
        assert files.attribution == "Vantor"


class TestStereoPlotterComposition:
    """StereoPlotter delegates discovery to a StereoFiles instance."""

    @pytest.fixture
    def stereo_plotter(self):
        return StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            dem_gsd=1,
            reference_dem="tests/test_data/ref_dem.tif",
        )

    def test_has_files_index(self, stereo_plotter):
        assert isinstance(stereo_plotter.files, StereoFiles)

    def test_properties_delegate_to_files(self, stereo_plotter):
        assert stereo_plotter.dem_fn == stereo_plotter.files.dem_fn
        assert stereo_plotter.full_directory == stereo_plotter.files.full_directory

    def test_attribution_passed_to_plotter(self, stereo_plotter):
        assert stereo_plotter.attribution == stereo_plotter.files.attribution


class TestStereoPlotter:
    @pytest.fixture
    def stereo_plotter(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            dem_gsd=1,
            reference_dem="tests/test_data/ref_dem.tif",
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_no_ref_dem(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            dem_gsd=1,
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_no_gsd(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            reference_dem="tests/test_data/ref_dem.tif",
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_dem_fn(self):
        stereo_plotter = StereoPlotter(
            directory="tests/test_data",
            stereo_directory="stereo",
            reference_dem="tests/test_data/ref_dem.tif",
            dem_fn="date_time_left_right_1m-DEM.tif",
            title="Stereo Results",
        )
        return stereo_plotter

    @pytest.fixture
    def stereo_plotter_without_intersection_error(self, stereo_plotter):
        stereo_plotter.files.intersection_error_fn = None
        return stereo_plotter

    def test_attribution_detection(self, stereo_plotter):
        """Test that StereoPlotter detects Vantor attribution from test data XMLs."""
        assert stereo_plotter.attribution == "Vantor"

    def test_plot_match_points(self, stereo_plotter):
        try:
            stereo_plotter.plot_match_points()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_disparity(self, stereo_plotter):
        try:
            stereo_plotter.plot_disparity(unit="meters")
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_dem_results(self, stereo_plotter):
        try:
            stereo_plotter.plot_dem_results()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_detailed_hillshade(self, stereo_plotter):
        try:
            stereo_plotter.plot_detailed_hillshade(subset_km=10)
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_detailed_hillshade_no_intersection_error(
        self, stereo_plotter_without_intersection_error
    ):
        try:
            stereo_plotter_without_intersection_error.plot_detailed_hillshade()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_instantiate_without_reference_dem(self, stereo_plotter_no_ref_dem):
        assert stereo_plotter_no_ref_dem.reference_dem is not None

    def test_instantiate_without_gsd(self, stereo_plotter_no_gsd):
        assert stereo_plotter_no_gsd.dem_fn is not None

    def test_instantiate_with_dem_fn(self, stereo_plotter_dem_fn):
        assert stereo_plotter_dem_fn.dem_fn is not None


class TestStereoPlotterNoMapproj:
    @pytest.fixture
    def stereo_plotter_no_mapproj(self):
        return StereoPlotter(
            directory="tests/test_data/no_mapproj",
            stereo_directory="stereo",
            title="Non-Mapproj Stereo Results",
        )

    @pytest.fixture
    def stereo_plotter_no_mapproj_no_intersection_error(
        self, stereo_plotter_no_mapproj
    ):
        stereo_plotter_no_mapproj.files.intersection_error_fn = None
        return stereo_plotter_no_mapproj

    def test_no_attribution_for_aster(self, stereo_plotter_no_mapproj):
        """Test that ASTER XMLs get no satellite attribution."""
        assert stereo_plotter_no_mapproj.attribution is None

    def test_orthos_false(self, stereo_plotter_no_mapproj):
        """Test that non-mapprojected data is correctly identified."""
        assert stereo_plotter_no_mapproj.orthos is False

    def test_alignment_matrices_loaded(self, stereo_plotter_no_mapproj):
        """Test that alignment matrix files are discovered."""
        assert stereo_plotter_no_mapproj.align_left_fn is not None
        assert stereo_plotter_no_mapproj.align_left_fn.endswith("run-align-L.txt")
        assert stereo_plotter_no_mapproj.align_right_fn is not None
        assert stereo_plotter_no_mapproj.align_right_fn.endswith("run-align-R.txt")

    def test_no_reference_dem_found(self, stereo_plotter_no_mapproj):
        """Test that missing reference DEM in logs is handled gracefully."""
        assert not stereo_plotter_no_mapproj.reference_dem

    def test_plot_match_points(self, stereo_plotter_no_mapproj):
        try:
            stereo_plotter_no_mapproj.plot_match_points()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_disparity(self, stereo_plotter_no_mapproj):
        try:
            stereo_plotter_no_mapproj.plot_disparity()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_dem_results(self, stereo_plotter_no_mapproj):
        try:
            stereo_plotter_no_mapproj.plot_dem_results()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_detailed_hillshade(self, stereo_plotter_no_mapproj):
        try:
            stereo_plotter_no_mapproj.plot_detailed_hillshade(subset_km=10)
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_detailed_hillshade_no_intersection_error(
        self, stereo_plotter_no_mapproj_no_intersection_error
    ):
        try:
            stereo_plotter_no_mapproj_no_intersection_error.plot_detailed_hillshade()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")


class TestStereoFilesMultiViewLayout:
    """ASP multi-view runs keep match files, sub-sampled scenes, and disparity
    in run-pair*/ subdirectories; the top level has only the joint products
    (L.tif, PC, DEM, IntersectionErr). StereoFiles must tolerate that layout
    and the plotters must fall back to "missing" placeholders (#160 tracks
    rendering the per-pair files)."""

    @pytest.fixture
    def mvs_directory(self, tmp_path):
        import shutil
        from pathlib import Path

        src = Path("tests/test_data/stereo")
        dst = tmp_path / "stereo"
        shutil.copytree(src, dst)
        # Strip the per-pair files a multi-view run keeps in run-pair*/:
        # match files, sub-sampled scenes, disparity, and the match CSV.
        for pattern in ["*.match", "*_sub.tif", "*-D.tif", "*-L__R.csv"]:
            for f in dst.glob(pattern):
                f.unlink()
        return str(tmp_path)

    @pytest.fixture
    def plotter(self, mvs_directory):
        return StereoPlotter(
            directory=mvs_directory,
            stereo_directory="stereo",
            dem_gsd=1,
            reference_dem="tests/test_data/ref_dem.tif",
            title="MVS Results",
        )

    def test_files_tolerate_missing_pairwise_products(self, plotter):
        assert plotter.files.match_point_fn is None
        assert plotter.files.disparity_sub_fn is None
        assert plotter.files.left_image_sub_fn is None
        assert plotter.files.right_image_sub_fn is None
        # The joint products are still discovered.
        assert plotter.files.dem_fn is not None
        assert plotter.files.intersection_error_fn is not None

    def test_plot_match_points_placeholder(self, plotter):
        try:
            plotter.plot_match_points()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_disparity_placeholder(self, plotter):
        try:
            plotter.plot_disparity()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_plot_dem_results_still_works(self, plotter):
        try:
            plotter.plot_dem_results()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
