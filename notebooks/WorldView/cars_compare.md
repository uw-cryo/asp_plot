# CARS vs ASP DEM Comparison — SpaceNet UCSD WorldView-3

## Goal

Compare DEMs produced by ASP and CARS on the same WorldView-3 stereo pair:

- **ASP** (NASA Ames Stereo Pipeline) — already processed in `worldview_spacenet_ucsd_stereo.ipynb`
- **CARS** (CNES open-source stereo pipeline) — to be processed

## About CARS

[CARS](https://github.com/CNES/cars) is an open-source Python tool developed by CNES (French Space Agency) for producing DSMs from satellite stereo imagery. Key features:

- **Dense matching**: Uses [Pandora](https://github.com/CNES/Pandora) (Census cost + SGM) with multiple presets for urban, mountain, vegetation
- **Multi-resolution**: Coarse-to-fine processing (default 16x → 4x → 1x) progressively narrows disparity search range
- **Geometry**: Uses [Shareloc](https://github.com/CNES/shareloc) for RPC handling — reads RPCs directly from GeoTIFF metadata
- **Tiled architecture**: Processes data in tiles with configurable parallelism and per-worker memory limits — avoids the full-scene memory issues encountered with SETSM
- **No bundle adjustment** in the standard pipeline. Instead, CARS corrects epipolar grids using sparse SIFT tie points. Separate `cars-bundleadjustment` tool available as an optional extra.

## Input Data

Same WorldView-3 stereo pair as the ASP/SETSM experiments:

| Catalog ID | Date | Off-nadir | Image size |
|---|---|---|---|
| 1040010007A3D100 | 2015-02-11 | 24.3° | 43008 x 44032 px |
| 1040010007A93700 | 2015-02-12 | 9.6° | 43008 x 46080 px |

Full images converted from NTF to GeoTIFF (with embedded RPCs) are in `ucsd_stereo_SETSM_full/` from the prior experiment. CARS reads RPCs directly from GeoTIFF metadata — no separate XML files needed.

## Processing Approach: ROI on Full Images

CARS supports two approaches for sub-scene processing:

1. **`cars-extractroi`** CLI tool — pre-crops images and writes adjusted RPCs as `.RPB` sidecar files
2. **`roi` config parameter** — uses full images for sparse matching / grid correction but only runs dense matching within the specified ROI

We use approach 2 (`roi`) because:
- Full images give CARS more context for SIFT tie-point matching and epipolar grid correction
- No manual RPC adjustment needed
- CARS handles tiling and memory internally

### ROI definition

The ROI matches the ASP processing crop area (UCSD campus region):

```
Lon: -117.262 to -117.219
Lat:  32.854 to  32.890
(~4 km x 4 km, UTM 11N: ~475500-479500 E, 3635100-3639100 N)
```

## Running CARS via Docker

### Docker image

```bash
docker pull cnes/cars
```

The image is based on Ubuntu 24 with OTB 9.1.1 and includes all dependencies (Pandora, Shareloc, rasterio with JP2OpenJPEG/NITF drivers).

### Configuration

CARS uses a single YAML config file. Minimal config with ROI:

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

### Run

```bash
docker run --platform linux/amd64 --rm \
    --cpus 6 \
    -v /path/to/ucsd_stereo_SETSM_full:/input:ro \
    -v /path/to/ucsd_stereo_CARS_crop:/output \
    cnes/cars /output/config.yaml
```

### Expected outputs

In `results/`:
- `dsm/dsm.tif` — DSM as GeoTIFF (float32)
- `dsm/image.tif` — Ortho-image
- `metadata.json` — Processing parameters and statistics
- `used_conf.json` — Full resolved configuration

### Notes

- **Vertical reference**: CARS defaults to EGM96 geoid heights. For comparison with ASP (ellipsoidal heights), may need to set `geoid: false` in the output config or convert afterwards.
- **Matching preset**: Default is `census_sgm_default` (Census5, P1=8, P2=32). For urban areas like UCSD, `census_sgm_urban` (Census11, P1=20, P2=80) may perform better.
- **Initial elevation**: Providing a coarse DEM via `input.initial_elevation.dem` narrows the disparity search range and speeds processing. The Copernicus 30m DEM at `ucsd_stereo/ref/cop30_ucsd_wgs84_utm.tif` could be used for this.

## Key Differences: CARS vs ASP vs SETSM

| Feature | ASP | CARS | SETSM |
|---|---|---|---|
| Language | C++ | Python + C++ (pybind11) | C++ |
| Dense matching | MGM/SGM (custom) | Pandora (Census + SGM) | Custom NCC |
| Bundle adjustment | Integrated | Separate tool (optional) | None (RPC bias only) |
| Memory management | In-process | Tiled, per-worker limits | Loads full images |
| Parallel model | OpenMP threads | multiprocessing / Dask | OpenMP / MPI |
| Cropped processing | mapproject + crop | ROI config or extractroi | Manual RPC surgery |
| Install | Binary download | pip / Docker | Build from source |

## References

- CARS repository: https://github.com/CNES/cars
- CARS documentation: https://cars.readthedocs.io
- Pandora (dense matching engine): https://github.com/CNES/Pandora
- Shareloc (geometry library): https://github.com/CNES/shareloc
- ASP documentation: https://stereopipeline.readthedocs.io
- SpaceNet UCSD dataset: https://spacenet.ai/core3d/
