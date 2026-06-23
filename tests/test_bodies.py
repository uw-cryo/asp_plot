import pytest

from asp_plot.bodies import BODIES, Body, body_for_dem


class TestBodyRegistry:
    def test_registry_keys(self):
        assert set(BODIES) == {"earth", "moon", "mars"}

    def test_entries_are_frozen_body_instances(self):
        for name, body in BODIES.items():
            assert isinstance(body, Body)
            assert body.name == name
            with pytest.raises(Exception):
                body.name = "changed"  # frozen dataclass

    def test_earth_facts(self):
        earth = BODIES["earth"]
        assert earth.altimetry_instrument == "ICESat-2"
        assert earth.iau_sphere_radius_m is None
        assert earth.datum is None
        assert earth.geocentric_proj is None  # EPSG:4978 used instead
        assert earth.geographic_crs_wkt is None
        assert earth.semi_major_axis_m == 6378137.0
        assert earth.inverse_flattening == pytest.approx(298.257223563)

    def test_moon_facts(self):
        moon = BODIES["moon"]
        assert moon.altimetry_instrument == "LOLA"
        assert moon.iau_sphere_radius_m == 1_737_400.0
        assert moon.datum == "D_MOON"
        assert moon.geocentric_proj == "+proj=geocent +R=1737400 +units=m +no_defs"
        assert "D_MOON" in moon.geographic_crs_wkt
        assert moon.semi_major_axis_m == 1_737_400.0
        assert moon.inverse_flattening == 0.0

    def test_mars_facts(self):
        mars = BODIES["mars"]
        assert mars.altimetry_instrument == "MOLA"
        assert mars.iau_sphere_radius_m == 3_396_190.0
        assert mars.datum == "D_MARS"
        assert mars.geocentric_proj == "+proj=geocent +R=3396190 +units=m +no_defs"
        assert "D_MARS" in mars.geographic_crs_wkt
        assert mars.semi_major_axis_m == 3_396_190.0
        assert mars.inverse_flattening == 0.0


class TestBodyForDem:
    def test_explicit_body_skips_detection(self):
        # Passing body= avoids any DEM read and returns the registry entry.
        assert body_for_dem("unused.tif", body="mars") is BODIES["mars"]
        assert body_for_dem("unused.tif", body="moon") is BODIES["moon"]
        assert body_for_dem("unused.tif", body="earth") is BODIES["earth"]
