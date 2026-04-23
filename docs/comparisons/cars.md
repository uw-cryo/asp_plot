# CARS vs ASP — SpaceNet UCSD WorldView-3

Compare DEMs produced by ASP and CARS on the same WorldView-3 stereo pair:

- **ASP** (NASA Ames Stereo Pipeline) — processed in the [UCSD notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo)
- **[CARS](https://github.com/CNES/cars)** (CNES open-source stereo pipeline)

## About CARS

CARS is an open-source Python tool developed by CNES (French Space Agency) for producing DSMs from satellite stereo imagery. Key features:

- **Dense matching**: Uses [Pandora](https://github.com/CNES/Pandora) (Census cost + SGM) with multiple presets for urban, mountain, vegetation
- **Multi-resolution**: Coarse-to-fine processing (default 16x → 4x → 1x) progressively narrows disparity search range
- **Geometry**: Uses [Shareloc](https://github.com/CNES/shareloc) for RPC handling — reads RPCs directly from GeoTIFF metadata
- **Tiled architecture**: Processes data in tiles with configurable parallelism and per-worker memory limits — avoids the full-scene memory issues encountered with SETSM
- **No bundle adjustment** in the standard pipeline. Instead, CARS corrects epipolar grids using sparse SIFT tie points. Separate `cars-bundleadjustment` tool available as an optional extra.

## Source Data

Same WorldView-3 stereo pair as the [SETSM comparison](setsm):

| Catalog ID | Date | Off-nadir | Image size |
|---|---|---|---|
| 1040010007A93700 | 2015-02-12 | 8.4° | 43008 × 46080 px |
| 1040010007CA4D00 | 2015-02-24 | 12.9° | 43008 × 46080 px |

This is the same pair (`21deg_12d`, convergence 21.2°) used in the [scene-selection notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo_scene_selection) and the [ASP processing notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo).

## Processing Approach: ROI on Full Images

CARS supports two approaches for sub-scene processing:

1. **`cars-extractroi`** — pre-crops images and writes adjusted RPCs as `.RPB` sidecar files
2. **`roi` config parameter** — uses full images for sparse matching / grid correction but only runs dense matching within the specified ROI

We use approach 2 (`roi`) because full images give CARS more context for SIFT tie-point matching and epipolar grid correction, with no manual RPC adjustment needed.

The ROI matches the ASP processing crop area (UCSD campus region, ~4 km × 4 km):

```
Lon: -117.262 to -117.219
Lat:  32.854 to  32.890
```

## Running CARS via Docker

```bash
docker pull cnes/cars

docker run --platform linux/amd64 --rm \
    --cpus 6 \
    -v /path/to/input_images:/input:ro \
    -v /path/to/output:/output \
    cnes/cars /output/config.yaml
```

:::{dropdown} Full configuration file (config.yaml)
:icon: file-code

```yaml
input:
  sensors:
    left:
      image: /input/1040010007A93700_P001.tif
    right:
      image: /input/1040010007CA4D00_P001.tif
  roi:
    type: FeatureCollection
    features:
      - type: Feature
        properties: {}
        geometry:
          type: Polygon
          coordinates:
            - - [-117.262, 32.854]
              - [-117.219, 32.854]
              - [-117.219, 32.890]
              - [-117.262, 32.890]
              - [-117.262, 32.854]

output:
  directory: /output/results
  resolution: 2.0
  epsg: 32611

orchestrator:
  mode: multiprocessing
  nb_workers: 2
  max_ram_per_worker: 1500
```

Key parameters:
- `output.resolution: 2.0` — 2 m output to match ASP processing
- `output.epsg: 32611` — UTM 11N (same CRS as ASP DEM)
- `orchestrator.nb_workers: 2` — limit parallelism for 16 GB laptop
- `orchestrator.max_ram_per_worker: 1500` — 1.5 GB per worker
:::

Expected outputs in `results/`:
- `dsm/dsm.tif` — DSM as GeoTIFF (float32)
- `dsm/image.tif` — Ortho-image
- `metadata.json` — Processing parameters and statistics

:::{dropdown} Notes on configuration
:icon: light-bulb

- **Vertical reference**: CARS defaults to EGM96 geoid heights. For comparison with ASP (ellipsoidal heights), set `geoid: false` in the output config or convert afterwards.
- **Matching preset**: Default is `census_sgm_default` (Census5, P1=8, P2=32). For urban areas, `census_sgm_urban` (Census11, P1=20, P2=80) may perform better.
- **Initial elevation**: Providing a coarse DEM via `input.initial_elevation.dem` narrows the disparity search range and speeds processing.
:::

## Run Metrics

| Metric | Value |
|---|---|
| Computation time | ~17 h 18 min |
| Peak memory | ~10.3 GB (hit CARS' `RAM available < 500 Mb` warnings near the end) |
| Docker `--memory` limit | 11 GB |
| Output DSM | 2017 × 2001 px, 2 m, EPSG:32611 + EGM96 |
| Output std dev (raw) | 134.6 m (inflated by edge blunders outside the 3 km comparison box) |
| Output std dev (3 km comparison crop) | 48.7 m |
| Convergence angle | 21.2° |

CARS is much slower than SETSM on this laptop (17 h vs 20 min), primarily because Pandora's per-tile Census + SGM matching is done in pure Python multiprocessing with two workers at 1.5 GB each. The run reports `RAM available < 500 Mb` warnings during the final `texture / dsm / dsm_weights` passes but completes successfully. The raw DSM carries some edge blunders (min −321 m, max 1273 m) outside the canonical 3 km comparison area; inside the comparison crop the range collapses to approximately −200 m to +200 m with outliers concentrated at the seaward edge.

## Hillshade Comparison

::::{grid} 3
:::{grid-item}
![Copernicus 30m](../examples/figures/ucsd-cop30m-hillshade.png)
**Copernicus 30m DEM**
:::
:::{grid-item}
![ASP 2m](../examples/figures/ucsd-asp2m-hillshade.png)
**ASP 2m DEM**
:::
:::{grid-item}
![CARS 2m](../examples/figures/ucsd-cars2m-hillshade.png)
**CARS 2m DEM**
:::
::::

CARS resolves the urban structure cleanly — buildings, streets, the I-5 freeway on the east, Mount Soledad terrain on the west — comparable to ASP in detail. Some speckle remains along the seaward edge where CARS' ROI extends beyond the stereo overlap. No tile-boundary artifacts.

## References

- [CARS repository](https://github.com/CNES/cars)
- [CARS documentation](https://cars.readthedocs.io)
- [Pandora dense matching engine](https://github.com/CNES/Pandora)
- [Shareloc geometry library](https://github.com/CNES/shareloc)
