import matplotlib
import pytest

from asp_plot.stereo_geometry import StereoGeometryPlotter
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser

matplotlib.use("Agg")


class TestStereoGeometryPlotter:
    @pytest.fixture
    def stereo_geometry_plotter(self):
        # add_basemap=False keeps the test offline; a True default would fetch
        # contextily tiles from a live server and hang/flake in CI.
        stereo_geometry_plotter = StereoGeometryPlotter(
            directory="tests/test_data",
            add_basemap=False,
        )
        return stereo_geometry_plotter

    @pytest.fixture
    def stereo_geometry_plotter_tiled(self):
        stereo_geometry_plotter = StereoGeometryPlotter(
            directory="tests/test_data/tiled_xmls",
            add_basemap=False,
        )
        return stereo_geometry_plotter

    def test_composes_parser_not_inherits(self, stereo_geometry_plotter):
        # The plotter should compose a parser rather than inherit from it
        assert not isinstance(stereo_geometry_plotter, StereopairMetadataParser)
        assert isinstance(stereo_geometry_plotter.parser, StereopairMetadataParser)

    def test_dg_geom_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_dg_geom_plot_tiled(self, stereo_geometry_plotter_tiled):
        try:
            stereo_geometry_plotter_tiled.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_satellite_position_orientation_plot(self, stereo_geometry_plotter):
        try:
            stereo_geometry_plotter.satellite_position_orientation_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_two_scene_single_figure(self, stereo_geometry_plotter, tmp_path):
        # Two scenes -> exactly one figure, named as given (unchanged behavior).
        saved = stereo_geometry_plotter.dg_geom_plot(
            save_dir=str(tmp_path), fig_fn="geom.png"
        )
        assert saved == ["geom.png"]
        assert (tmp_path / "geom.png").exists()


class TestStereoGeometryPlotterNScene:
    """More than two scenes: overview figure + one figure per pair."""

    THREE_SCENES = [
        "tests/test_data/10300100D0772D00.r100.xml",
        "tests/test_data/10300100D12D7400.r100.xml",
        "tests/test_data/tiled_xmls/10200100A1865800.r100.xml",
    ]

    @pytest.fixture
    def plotter_n(self):
        return StereoGeometryPlotter(inputs=self.THREE_SCENES, add_basemap=False)

    def test_emits_overview_plus_one_figure_per_pair(self, plotter_n, tmp_path):
        # 3 scenes -> 1 overview + 3 pair figures (3-choose-2).
        saved = plotter_n.dg_geom_plot(save_dir=str(tmp_path), fig_fn="geom.png")
        assert len(saved) == 4
        assert "geom_overview.png" in saved
        pair_files = [f for f in saved if f != "geom_overview.png"]
        assert len(pair_files) == 3
        for fn in saved:
            assert (tmp_path / fn).exists()

    def test_pair_filenames_keyed_by_catid(self, plotter_n, tmp_path):
        saved = plotter_n.dg_geom_plot(save_dir=str(tmp_path), fig_fn="geom.png")
        catids = ["10300100D0772D00", "10300100D12D7400", "10200100A1865800"]
        pair_files = [f for f in saved if f != "geom_overview.png"]
        # Every pair file is named with both scenes' CATIDs (no pairN fallback,
        # since these scenes all carry CATIDs).
        for fn in pair_files:
            present = [c for c in catids if c in fn]
            assert len(present) == 2

    def test_single_scene_raises(self, tmp_path):
        plotter = StereoGeometryPlotter(
            inputs=["tests/test_data/10300100D0772D00.r100.xml"], add_basemap=False
        )
        with pytest.raises(ValueError, match="at least two scenes"):
            plotter.dg_geom_plot(save_dir=str(tmp_path), fig_fn="geom.png")


class TestStereoGeometryPlotterPleiades:
    """Pléiades DIMAP inputs: no scan/TDI in labels, no covariance in plots."""

    @pytest.fixture
    def plotter(self):
        return StereoGeometryPlotter(
            directory="tests/test_data/pleiades",
            add_basemap=False,
        )

    def test_scene_string_omits_scan_and_tdi(self, plotter):
        p = plotter.parser.get_pair_dict()
        scene_string = plotter.get_scene_string(p)
        assert "ID:" in scene_string and "GSD:" in scene_string
        # DIMAP has no scan direction or TDI level; the label omits them
        # instead of printing scan:None / crashing on tdi=%i.
        assert "scan:" not in scene_string
        assert "tdi:" not in scene_string

    def test_dg_geom_plot(self, plotter):
        try:
            plotter.dg_geom_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")

    def test_satellite_position_orientation_plot_without_covariance(self, plotter):
        # DIMAP provides no ephemeris/attitude covariance; the plot must fall
        # back to plain position markers and an annotated covariance panel.
        try:
            plotter.satellite_position_orientation_plot()
        except Exception as e:
            pytest.fail(f"figure method raised an exception: {str(e)}")
