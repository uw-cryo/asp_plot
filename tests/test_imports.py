import pytest


class TestImports:
    def test_import(self):
        import asp_plot

    def test_import_asp_plot_modules(self):
        import asp_plot.utils
        import asp_plot.stereopair_metadata_parser
        import asp_plot.scenes
        import asp_plot.processing_parameters
        import asp_plot.bundle_adjust
        import asp_plot.scenes

    def test_import_asp_plot_classes(self):
        from asp_plot.utils import ColorBar, Raster, Plotter
        from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
        from asp_plot.scenes import ScenePlotter, SceneGeometryPlotter
        from asp_plot.processing_parameters import ProcessingParameters
        from asp_plot.bundle_adjust import ReadResiduals, PlotResiduals
        from asp_plot.stereo import StereoPlotter
