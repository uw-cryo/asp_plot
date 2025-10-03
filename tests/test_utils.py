import os

import numpy as np
import pytest

from asp_plot.utils import ColorBar, Raster


class TestRaster:
    """Test Raster class functionality."""

    @pytest.fixture
    def test_dem(self):
        """Path to test DEM file."""
        return "tests/test_data/stereo/date_time_left_right_1m-DEM.tif"

    @pytest.fixture
    def test_ref_dem(self):
        """Path to test reference DEM file."""
        return "tests/test_data/ref_dem.tif"

    def test_raster_init(self, test_dem):
        """Test basic Raster initialization."""
        raster = Raster(test_dem)
        assert raster.fn == test_dem
        assert raster.ds is not None
        assert raster.downsample == 1

    def test_raster_init_with_downsample(self, test_dem):
        """Test Raster initialization with downsampling."""
        raster = Raster(test_dem, downsample=5)
        assert raster.downsample == 5

    def test_data_property_lazy_loading(self, test_dem):
        """Test that data property lazy loads."""
        raster = Raster(test_dem)
        # Data should not be loaded yet
        assert raster._data is None
        # Accessing data should trigger loading
        data = raster.data
        assert data is not None
        assert isinstance(data, np.ma.MaskedArray)
        # Second access should return cached data
        data2 = raster.data
        assert data is data2  # Same object reference

    def test_read_array(self, test_dem):
        """Test read_array method."""
        raster = Raster(test_dem)
        data = raster.read_array()
        assert isinstance(data, np.ma.MaskedArray)
        assert data.ndim == 2

    def test_read_array_with_downsample(self, test_dem):
        """Test read_array with downsampling."""
        raster_full = Raster(test_dem, downsample=1)
        raster_ds = Raster(test_dem, downsample=2)

        data_full = raster_full.read_array()
        data_ds = raster_ds.read_array()

        # Downsampled should be smaller
        assert data_ds.shape[0] <= data_full.shape[0] / 2 + 1
        assert data_ds.shape[1] <= data_full.shape[1] / 2 + 1

    def test_transform_property(self, test_dem):
        """Test transform property."""
        raster = Raster(test_dem)
        transform = raster.transform
        assert transform is not None
        # Should be the original transform before data is loaded
        assert transform == raster.ds.transform

    def test_transform_updates_with_downsample(self, test_dem):
        """Test that transform updates when downsampling."""
        raster = Raster(test_dem, downsample=3)
        original_transform = raster.ds.transform
        # Trigger data loading which updates transform
        _ = raster.data
        updated_transform = raster.transform
        # Resolution should increase with downsampling
        assert updated_transform[0] > original_transform[0]

    def test_get_epsg_code(self, test_dem):
        """Test getting EPSG code."""
        raster = Raster(test_dem)
        epsg = raster.get_epsg_code()
        assert isinstance(epsg, int)

    def test_get_gsd(self, test_dem):
        """Test getting ground sample distance."""
        raster = Raster(test_dem)
        gsd = raster.get_gsd()
        assert isinstance(gsd, (int, float))
        assert gsd > 0

    def test_get_ndv(self, test_dem):
        """Test getting nodata value."""
        raster = Raster(test_dem)
        ndv = raster.get_ndv()
        assert ndv is not None

    def test_hillshade(self, test_dem):
        """Test hillshade generation."""
        raster = Raster(test_dem)
        hillshade = raster.hillshade()
        assert isinstance(hillshade, np.ma.MaskedArray)
        assert hillshade.ndim == 2

    def test_load_and_diff_rasters(self, test_dem, test_ref_dem):
        """Test static method for loading and differencing rasters."""
        diff, transform, crs, nodata = Raster.load_and_diff_rasters(
            test_dem, test_ref_dem
        )
        assert isinstance(diff, np.ma.MaskedArray)
        assert diff.ndim == 2
        assert transform is not None
        assert crs is not None
        assert nodata == -9999
        # Should have both positive and negative values for a real difference
        assert np.any(diff > 0) or np.any(diff < 0)

    def test_compute_difference_no_save(self, test_dem, test_ref_dem, tmp_path):
        """Test compute_difference without saving."""
        # Use tmp_path to avoid modifying test data
        import shutil

        temp_dem = tmp_path / "test_dem.tif"
        shutil.copy(test_dem, temp_dem)

        raster = Raster(str(temp_dem))
        diff = raster.compute_difference(test_ref_dem, save=False)
        assert isinstance(diff, np.ma.MaskedArray)
        # Should not create a diff file
        diff_file = str(temp_dem).replace(".tif", "_ref_dem_diff.tif")
        assert not os.path.exists(diff_file)

    def test_compute_difference_with_save(self, test_dem, test_ref_dem, tmp_path):
        """Test compute_difference with saving."""
        # Copy test DEM to tmp directory to avoid polluting test data
        import shutil

        temp_dem = tmp_path / "test_dem.tif"
        shutil.copy(test_dem, temp_dem)

        raster = Raster(str(temp_dem))
        diff = raster.compute_difference(test_ref_dem, save=True)

        # Check that diff file was created
        expected_diff_file = str(temp_dem).replace(
            "test_dem.tif", "test_dem_ref_dem_diff.tif"
        )
        assert os.path.exists(expected_diff_file)
        assert isinstance(diff, np.ma.MaskedArray)

    def test_save_raster(self, test_dem, tmp_path):
        """Test static save_raster method."""
        raster = Raster(test_dem)
        data = raster.read_array()

        # Modify data slightly
        modified_data = data * 2

        output_file = tmp_path / "modified.tif"
        Raster.save_raster(
            modified_data, str(output_file), reference_fn=test_dem, dtype=np.float32
        )

        # Check file was created and can be read
        assert output_file.exists()
        saved_raster = Raster(str(output_file))
        saved_data = saved_raster.read_array()
        assert saved_data.shape == modified_data.shape


class TestColorBar:
    """Test ColorBar class functionality."""

    def test_colorbar_init(self):
        """Test ColorBar initialization."""
        cb = ColorBar(perc_range=(5, 95), symm=True)
        assert cb.perc_range == (5, 95)
        assert cb.symm is True
        assert cb.clim is None

    def test_get_clim(self):
        """Test color limit calculation."""
        cb = ColorBar(perc_range=(10, 90))
        data = np.random.randn(100, 100)
        clim = cb.get_clim(data)
        assert len(clim) == 2
        assert clim[0] < clim[1]

    def test_get_clim_symmetric(self):
        """Test symmetric color limits."""
        cb = ColorBar(perc_range=(10, 90), symm=True)
        data = np.random.randn(100, 100)
        clim = cb.get_clim(data)
        assert clim[0] == -clim[1]  # Should be symmetric

    def test_find_common_clim(self):
        """Test finding common limits across multiple arrays."""
        cb = ColorBar()
        data1 = np.random.randn(50, 50)
        data2 = np.random.randn(50, 50) * 2
        clim = cb.find_common_clim([data1, data2])
        assert len(clim) == 2
        assert clim[0] < clim[1]

    def test_get_cbar_extend(self):
        """Test colorbar extension determination."""
        cb = ColorBar()
        data = np.random.randn(100, 100)
        cb.get_clim(data)
        extend = cb.get_cbar_extend(data)
        assert extend in ["neither", "min", "max", "both"]

    def test_get_norm(self):
        """Test normalization."""
        cb = ColorBar()
        data = np.random.randn(100, 100)
        cb.get_clim(data)
        norm = cb.get_norm(lognorm=False)
        assert norm is not None
