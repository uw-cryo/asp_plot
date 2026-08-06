"""Tests for the ground-control comparison prototype (issue #156).

The fixture parquet is synthesized against the committed ``ref_dem.tif`` in
the DEM's own CRS, so nothing here needs the groundcontrol package; the
transform-path tests only assert the fail-loud behaviours.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import Point

from asp_plot.altimetry_source import AltimetrySource
from asp_plot.control_points import ControlPoints, ControlPointsPlotter, decimal_year

matplotlib.use("Agg")

DEM = str(Path(__file__).parent / "test_data" / "ref_dem.tif")
DH_TRUTH = 0.5  # control heights are seeded DEM - 0.5, so DEM - control = +0.5


@pytest.fixture
def control_parquet(tmp_path):
    """Synthetic control points on the ref_dem grid, already in the DEM CRS."""
    dem = AltimetrySource._open_dem(DEM)
    xs = dem.x.values[[1, 2, 3]]
    ys = dem.y.values[[1, 2, 3]]
    pts = [Point(x, y) for x in xs for y in ys]
    sampled = dem.interp(
        x=("z", [p.x for p in pts]), y=("z", [p.y for p in pts])
    ).values
    noise = np.random.default_rng(42).normal(0, 0.05, len(pts))
    gdf = gpd.GeoDataFrame(
        {
            "source": ["ngs"] * len(pts),
            "height": sampled - DH_TRUTH + noise,
            "raw": [json.dumps({"vertSource": "ADJUSTED"})] * len(pts),
        },
        geometry=pts,
        crs=dem.rio.crs,
    )
    fn = tmp_path / "control.parquet"
    gdf.to_parquet(fn)
    return str(fn)


class TestFilterQuality:
    def test_drops_low_grade_and_heightless_rows(self):
        gdf = gpd.GeoDataFrame(
            {
                "source": ["ngs", "ngs", "ngs", "3dep", "3dep", "opus", "ngl"],
                "height": [10.0, 10.0, np.nan, 10.0, 10.0, 10.0, 10.0],
                "raw": [
                    json.dumps({"vertSource": "ADJUSTED"}),
                    json.dumps({"vertSource": "SCALED"}),
                    json.dumps({"vertSource": "ADJUSTED"}),
                    json.dumps({}),
                    json.dumps({}),
                    json.dumps({}),
                    json.dumps({}),
                ],
                "point_type": ["monument"] * 3 + ["NVA", "BVA", "gnss", "gnss"],
            },
            geometry=[Point(0, i) for i in range(7)],
            crs="EPSG:4326",
        )
        out = ControlPoints.filter_quality(gdf)
        # kept: survey-grade ngs, topo 3dep checkpoint, pass-through opus;
        # dropped: scaled/heightless ngs, bathy 3dep, antenna-reference ngl
        assert list(out.index) == [0, 3, 5]

    def test_bare_gdf_passes_through(self):
        gdf = gpd.GeoDataFrame(
            {"height": [1.0, np.nan]},
            geometry=[Point(0, 0), Point(0, 1)],
            crs="EPSG:4326",
        )
        assert len(ControlPoints.filter_quality(gdf)) == 1


class TestSampleDem:
    def test_recovers_seeded_offset(self, control_parquet):
        cp = ControlPoints(control_parquet, DEM)
        pts = cp.sample_dem(n_sigma=None)
        assert cp.stats["n"] == 9
        assert cp.stats["median"] == pytest.approx(DH_TRUTH, abs=0.1)
        assert cp.stats["nmad"] == pytest.approx(0.05, abs=0.1)
        assert "dem_minus_control" in pts.columns

    def test_outlier_filter_drops_blunder(self, control_parquet, tmp_path):
        gdf = gpd.read_parquet(control_parquet)
        gdf.loc[gdf.index[0], "height"] -= 1000.0  # gross blunder
        fn = tmp_path / "with_blunder.parquet"
        gdf.to_parquet(fn)
        cp = ControlPoints(str(fn), DEM)
        cp.sample_dem(n_sigma=3)
        # a 3-sigma *std* filter could never reject at n=9 (max |z| = 8/3);
        # the NMAD filter must.
        assert cp.stats["n"] == 8
        assert cp.stats["median"] == pytest.approx(DH_TRUTH, abs=0.1)


class TestToDemFrame:
    def test_matching_crs_skips_transform(self, control_parquet):
        cp = ControlPoints(control_parquet, DEM)  # no epoch given
        out = cp.to_dem_frame()
        assert len(out) == 9

    def test_crs_mismatch_without_epoch_raises(self, control_parquet):
        gdf = gpd.read_parquet(control_parquet).to_crs("EPSG:4326")
        cp = ControlPoints(control_parquet, DEM)
        with pytest.raises(ValueError, match="epoch"):
            cp.to_dem_frame(gdf)

    def test_missing_groundcontrol_is_actionable(self, control_parquet, monkeypatch):
        gdf = gpd.read_parquet(control_parquet).to_crs("EPSG:4326")
        cp = ControlPoints(control_parquet, DEM, epoch=2020.0)
        monkeypatch.setitem(sys.modules, "groundcontrol", None)
        monkeypatch.setitem(sys.modules, "groundcontrol.crs", None)
        with pytest.raises(ImportError, match="groundcontrol"):
            cp.to_dem_frame(gdf)


class TestPlot:
    def test_saves_figure(self, control_parquet, tmp_path):
        cp = ControlPoints(control_parquet, DEM)
        plotter = ControlPointsPlotter(cp)
        plotter.plot_control_dh(save_dir=str(tmp_path), fig_fn="ctrl.png")
        assert (tmp_path / "ctrl.png").exists()


def test_decimal_year():
    from datetime import datetime

    assert decimal_year(datetime(2010, 1, 1)) == pytest.approx(2010.0)
    assert decimal_year(datetime(2009, 12, 22)) == pytest.approx(2009.9726, abs=1e-3)
