import pytest


class TestImports:
    def test_import(self):
        import asp_plot

    def test_import_asp_plot_modules(self):
        import asp_plot.utils
        import asp_plot.scenes
        import asp_plot.processing_parameters
        import asp_plot.bundle_adjust

    def test_import_asp_plot_classes(self):
        from asp_plot.utils import ColorBar, Raster, Plotter
        from asp_plot.scenes import ScenePlotter
        from asp_plot.processing_parameters import ProcessingParameters
        from asp_plot.bundle_adjust import ReadResiduals, PlotResiduals
