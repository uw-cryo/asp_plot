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
| 1040010007A3D100 | 2015-02-11 | 24.3° | 43008 x 44032 px |
| 1040010007A93700 | 2015-02-12 | 9.6° | 43008 x 46080 px |

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
      image: /input/1040010007A3D100_P001.tif
    right:
      image: /input/1040010007A93700_P001.tif
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
  nb_workers: 4
  max_ram_per_worker: 2000
```

Key parameters:
- `output.resolution: 2.0` — 2 m output to match ASP processing
- `output.epsg: 32611` — UTM 11N (same CRS as ASP DEM)
- `orchestrator.nb_workers: 4` — limit parallelism for 16 GB laptop
- `orchestrator.max_ram_per_worker: 2000` — 2 GB per worker (CARS default)
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
<!-- Add ucsd-cars2m-hillshade.png to docs/_static/comparisons/ when available -->
*CARS 2m DEM — pending*
:::
::::

## References

- [CARS repository](https://github.com/CNES/cars)
- [CARS documentation](https://cars.readthedocs.io)
- [Pandora dense matching engine](https://github.com/CNES/Pandora)
- [Shareloc geometry library](https://github.com/CNES/shareloc)
