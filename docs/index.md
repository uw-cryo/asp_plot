# asp_plot

[![PyPI](https://img.shields.io/pypi/v/asp-plot.svg)](https://pypi.org/project/asp-plot/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/asp-plot.svg)](https://anaconda.org/conda-forge/asp-plot)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14263121.svg)](https://doi.org/10.5281/zenodo.14263121)
[![RTD](https://readthedocs.org/projects/asp-plot/badge/?version=latest)](https://asp-plot.readthedocs.io/en/latest/)

A Python package for visualizing output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

`asp_plot` generates diagnostic plots and comprehensive PDF reports for ASP stereo processing results, similar to reports from commercial SfM software like Agisoft Metashape.

## Features

::::{grid} 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

Install via conda or pip.
:::

:::{grid-item-card} CLI Usage
:link: cli/index
:link-type: doc

Generate reports from the command line.
:::

:::{grid-item-card} Example Reports
:link: examples/reports
:link-type: doc

View PDF reports for different sensors.
:::

:::{grid-item-card} Example Notebooks
:link: examples/index
:link-type: doc

Modular usage examples by sensor type.
:::

:::{grid-item-card} API Reference
:link: autoapi/index
:link-type: doc

Full Python API documentation.
:::

:::{grid-item-card} Contributing
:link: contributing
:link-type: doc

Development setup, testing, and releases.
:::

::::

## Supported Sensors

- **Earth-based**: WorldView, Pléiades, ASTER
- **Lunar**: Lunar Reconnaissance Orbiter Narrow Angle Camera (LRO NAC)
- **Mars**: Mars Reconnaissance Orbiter CTX and HiRISE, Mars Global Surveyor MOC

### Stereo-geometry metadata support

PDF reports work for any ASP processing directory — when a sensor's camera
metadata is not supported, the stereo geometry section is skipped gracefully
and everything else still renders. The table below is specifically about the
camera-metadata readers behind the stereo geometry plots (`stereo_geom` and
the report's Stereo Geometry section); see
[uw-cryo/asp_plot#168](https://github.com/uw-cryo/asp_plot/issues/168) for the
expansion roadmap.

| Sensor | ASP session | Camera metadata format | Status |
|---|---|---|---|
| WorldView / GeoEye / QuickBird / IKONOS | `dg` | DigitalGlobe XML | ✅ Supported (validated with real data) |
| Pléiades Neo | `pleiades` | DIMAP v2 (`PNEO_SENSOR`) | ✅ Supported (validated with real data) |
| Pléiades 1A/1B | `pleiades` | DIMAP v2 (`PHR_SENSOR`) | ✅ Supported (polynomial attitude implemented from the ASP reader spec) |
| SPOT 6/7 | `pleiades` | DIMAP v2 (`S6_SENSOR`/`S7_SENSOR`) | 🧪 Implemented from the ASP reader spec, not yet validated with real data — reports welcome ([open an issue](https://github.com/uw-cryo/asp_plot/issues/new)) |
| PeruSat-1 | `perusat` | DIMAP v2 (`PER1_SENSOR`) | 🧪 Implemented from the ASP reader spec, not yet validated with real data — reports welcome ([open an issue](https://github.com/uw-cryo/asp_plot/issues/new)) |
| SPOT 5 | `spot5` | DIMAP v1 | 🧪 Implemented from the ASP reader spec, not yet validated with real data — reports welcome ([open an issue](https://github.com/uw-cryo/asp_plot/issues/new)) |
| ALOS PRISM | `prism` | DIMAP-like (`ALOS`) | 🧪 Implemented from the ASP reader spec, not yet validated with real data — reports welcome ([open an issue](https://github.com/uw-cryo/asp_plot/issues/new)) |
| ASTER | `aster` | `gen_aster` XML | ✅ Supported (geometry derived from look vectors; no attitude, sun angles, or timestamps exist) |
| Cartosat-1, Deimos, and other RPC-only products | `rpc` | RPC coefficients in the image or a `*_RPC.TXT` sidecar | ✅ Supported (geometry derived from the RPCs and validated against vendor-reported view angles, GSD and footprints; no timestamps, attitude, sun angles, or along/across-track split exist) |
| Planetary (LRO NAC, CTX, HiRISE, MOC, ...) | `csm` / ISIS | CSM model state (JSON) | Handled by the CSM camera modules (`csm_camera_plot`), not the sensor readers |

Sensors report attitude either as quaternions (WorldView, DIMAP v2), as
roll/pitch/yaw angles (DIMAP v1), or not at all (ASTER). For the first group the
orientation plot shows roll/pitch/yaw computed relative to the orbital frame;
for the second it shows the vendor's angles as delivered, and the panel title
names the frame they are defined in — SPOT 5's are in the SPOT Geometry
Handbook navigation frame, which is *not* the same axis convention as the
computed ones. DIMAP v1 also reports no satellite azimuth, so the skyplot and
the pair convergence angle are unavailable for SPOT 5 and ALOS PRISM.

Two readers *derive* their geometry instead of parsing it. ASTER's `gen_aster`
camera files record only satellite positions and per-pixel look vectors, so the
footprint, view angles and ground sample distance are computed by intersecting
those look rays with the WGS84 ellipsoid. That gives a full skyplot and
convergence angle (the derivation reproduces ASTER's published 27.6° backward
pointing and ~0.6 base-to-height ratio), but no attitude panel, no sun angles,
and no timestamps — the acquisition date is recovered from a neighbouring
`AST_L1A_*` granule name when one is present, and is then shared by both bands
of the pair.

RPC-only products go one step further: there is no camera *file* at all, only
rational polynomial coefficients inside the image (or a `*_RPC.TXT` sidecar,
including Cartosat-1's `*_RPC_ORG.TXT` variant). Projecting a pixel to the
ground at two heights traces its look ray, which yields the footprint, satellite
azimuth/elevation, and — by intersecting the look rays from opposite ends of an
image line — the satellite position and off-nadir angle. Point `stereo_geom` at
the images themselves rather than at XMLs. Attitude, timestamps (unless the
image header records one), sun angles and the along/across-track split of the
off-nadir angle do not exist for these products and are reported as not
provided.

## What it does

- Stereo DEM processing visualization (hillshades, disparity maps, match points)
- Bundle adjustment analysis (residual maps, histograms)
- CSM camera model comparisons (position/orientation differences)
- ICESat-2 ATL06-SR altimetry comparisons (Earth-based only), with optional automatic `pc_align` refinement and a before/after alignment report
- Stereo geometry visualization from satellite XML metadata
- Comprehensive PDF report generation

```{toctree}
:maxdepth: 2
:hidden:

installation
cli/index
examples/index
examples/reports
API Reference <autoapi/index>
contributing
changelog
```
