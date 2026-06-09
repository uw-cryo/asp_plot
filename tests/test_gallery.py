import os

import matplotlib
import pytest

from asp_plot.gallery import GalleryPlotter

matplotlib.use("Agg")


# Three small synthetic DEMs already in the repo, with deliberately different
# elevation ranges so the shared-clim test is meaningful.
DEMS = [
    "tests/test_data/ref_dem.tif",
    "tests/test_data/stereo/date_time_left_right_1m-DEM.tif",
    "tests/test_data/no_mapproj/stereo/run-DEM.tif",
]


class TestGalleryPlotter:
    @pytest.fixture
    def gallery(self):
        return GalleryPlotter(DEMS, downsample=1)

    def test_empty_raster_list_raises(self):
        with pytest.raises(ValueError):
            GalleryPlotter([])

    def test_from_directory(self):
        gallery = GalleryPlotter.from_directory(
            "tests/test_data/stereo", pattern="*-DEM.tif"
        )
        assert gallery.raster_list == [
            "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"
        ]

    def test_from_directory_no_match_raises(self):
        with pytest.raises(ValueError):
            GalleryPlotter.from_directory(
                "tests/test_data", pattern="*-does-not-exist.tif"
            )

    @pytest.mark.parametrize(
        "n,expected",
        [(1, (1, 1)), (4, (2, 2)), (7, (3, 3))],
    )
    def test_grid_shape(self, n, expected):
        # Default figsize is (7.5, 10.0).
        assert GalleryPlotter._grid_shape(n, 7.5, 10.0) == expected

    def test_shared_clim(self, gallery):
        from asp_plot.utils import ColorBar, Raster

        arrays = [Raster(fn).read_array() for fn in DEMS]
        clim = gallery.cb.find_common_clim(arrays)
        # The shared clim aggregates each raster's (percentile) clim: it must be
        # at least as wide as every individual raster's own clim.
        per_clims = [ColorBar().get_clim(a) for a in arrays]
        assert clim[0] <= min(c[0] for c in per_clims) + 1e-6
        assert clim[1] >= max(c[1] for c in per_clims) - 1e-6
        assert clim[0] < clim[1]

    def test_plot_gallery_dem(self, gallery):
        fig = gallery.plot_gallery(hillshade=True)
        # Active (non-off) axes should equal the number of rasters.
        active = [ax for ax in fig.axes if ax.axison]
        # fig.axes also includes the colorbar axis, so filter to the panel grid
        # by checking each has a title matching one of our filenames.
        titled = [
            ax for ax in active if ax.get_title() and ax.get_title().endswith(".tif")
        ]
        assert len(titled) == len(DEMS)

    def test_plot_gallery_no_hillshade(self, gallery):
        try:
            gallery.plot_gallery(hillshade=False)
        except Exception as e:
            pytest.fail(f"plot_gallery(hillshade=False) raised: {e}")

    def test_save_output(self, gallery, tmp_path):
        gallery.plot_gallery(save_dir=str(tmp_path), fig_fn="test_gallery.png")
        assert os.path.exists(os.path.join(str(tmp_path), "test_gallery.png"))


class TestGalleryCLI:
    def test_cli_smoke(self, tmp_path):
        from click.testing import CliRunner

        from asp_plot.cli.gallery import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            DEMS
            + [
                "--output_directory",
                str(tmp_path),
                "--output_filename",
                "cli_gallery.png",
            ],
        )
        assert result.exit_code == 0, result.output
        assert os.path.exists(os.path.join(str(tmp_path), "cli_gallery.png"))
