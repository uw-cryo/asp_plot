class TestImports:
    def test_import(self):
        import asp_plot

        assert asp_plot is not None

    def test_import_asp_plot_modules(self):
        import asp_plot.alignment
        import asp_plot.altimetry
        import asp_plot.bundle_adjust
        import asp_plot.csm_camera
        import asp_plot.processing_parameters
        import asp_plot.scenes
        import asp_plot.stereo_geometry
        import asp_plot.stereopair_metadata_parser
        import asp_plot.utils

        assert asp_plot.stereo_geometry is not None
        assert asp_plot.utils is not None
        assert asp_plot.stereopair_metadata_parser is not None
        assert asp_plot.scenes is not None
        assert asp_plot.processing_parameters is not None
        assert asp_plot.csm_camera is not None
        assert asp_plot.bundle_adjust is not None
        assert asp_plot.altimetry is not None
        assert asp_plot.alignment is not None

    def test_import_asp_plot_classes(self):
        from asp_plot.alignment import Alignment
        from asp_plot.altimetry import Altimetry
        from asp_plot.bundle_adjust import PlotBundleAdjustFiles, ReadBundleAdjustFiles
        from asp_plot.processing_parameters import ProcessingParameters
        from asp_plot.scenes import ScenePlotter
        from asp_plot.stereo import StereoPlotter
        from asp_plot.stereo_geometry import StereoGeometryPlotter
        from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
        from asp_plot.utils import ColorBar, Plotter, Raster

        assert StereoGeometryPlotter is not None
        assert ColorBar is not None
        assert Raster is not None
        assert Plotter is not None
        assert StereopairMetadataParser is not None
        assert ScenePlotter is not None
        assert ProcessingParameters is not None
        assert ReadBundleAdjustFiles is not None
        assert PlotBundleAdjustFiles is not None
        assert StereoPlotter is not None
        assert Altimetry is not None
        assert Alignment is not None
