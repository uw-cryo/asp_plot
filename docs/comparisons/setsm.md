# SETSM vs ASP — SpaceNet UCSD WorldView-3

Compare DEMs produced by two open-source stereo photogrammetry pipelines on the same WorldView-3 stereo pair:

- **ASP** (NASA Ames Stereo Pipeline) — processed in the [UCSD notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo)
- **[SETSM](https://github.com/setsmdeveloper/SETSM)** (Surface Extraction with TIN-based Search-space Minimization) — Ohio State / Polar Geospatial Center

Both pipelines accept raw satellite imagery with RPC camera models and produce gridded DSMs.

## Source Data

IARPA CORE3D SpaceNet UCSD dataset — two WorldView-3 panchromatic acquisitions:

| Catalog ID | Date | Off-nadir | Image size |
|---|---|---|---|
| 1040010007A3D100 | 2015-02-11 | 24.3° | 43008 x 44032 px |
| 1040010007A93700 | 2015-02-12 | 9.6° | 43008 x 46080 px |

## Approach

SETSM is open-source (Apache 2.0), pure C++ with no GPU requirement. We built and ran it via Docker to avoid host dependency conflicts.

SETSM works on raw (non-mapprojected) imagery with RPC camera models read from DigitalGlobe XML metadata files. The full images are ~800 MB each (2.6 GB + 2.7 GB on disk as GeoTIFF), so memory becomes the limiting factor on a laptop:

- **Without sub-scene flags:** SETSM builds image pyramids for the entire scene by default. Two UInt16 images plus pyramids exceed 14 GB RAM and the container is OOM-killed (exit code 137) on a 16 GB machine.
- **With `-boundary_*` flags:** SETSM only reads and pyramids the subset of each image needed for the requested output extent, so memory scales with the output area, not the input image size.

:::{dropdown} Docker build details
:icon: terminal

```bash
git clone https://github.com/setsmdeveloper/SETSM.git
cd SETSM

docker build --platform linux/amd64 -t setsm -f- . <<'DOCKERFILE'
FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ make git \
    libgeotiff-dev libtiff-dev libproj-dev libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
COPY . /opt/SETSM
WORKDIR /opt/SETSM
RUN make INCS="-I/usr/include/geotiff"

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeotiff5 libgomp1 libtiff6 libproj25 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/SETSM/setsm /usr/local/bin/setsm
COPY --from=builder /opt/SETSM/default.txt /usr/local/share/setsm/default.txt
WORKDIR /data
ENTRYPOINT ["sh", "-c", "cp /usr/local/share/setsm/default.txt /data/default.txt 2>/dev/null; exec setsm \"$@\"", "--"]
DOCKERFILE
```

Multi-stage build — the final image contains only the `setsm` binary and runtime libraries. The `default.txt` config file is bundled and auto-copied at runtime (SETSM requires it in the working directory).
:::

## Run

```bash
docker run --platform linux/amd64 --rm --name setsm_boundary \
    --memory 11g --cpus 6 \
    -v /path/to/data:/data -w /data setsm \
    -image 1040010007A3D100_P001.tif -image 1040010007A93700_P001.tif \
    -outpath /data/results -outres 2 -mem 10 -minH 0 -maxH 300 \
    -boundary_min_X 476000 -boundary_min_Y 3635600 \
    -boundary_max_X 479000 -boundary_max_Y 3638600 \
    -projection utm
```

The boundary defines a 3 km × 3 km UTM 11N area matching the ASP processing extent.

| Metric | Value |
|---|---|
| Computation time | ~20 min |
| Peak memory (SETSM-reported) | 2.28 GB |
| Docker `--memory` limit | 11 GB |
| Output DEM | 1500 × 1500 px, 2 m, EPSG:32611 |
| Output std dev | 35.2 m |
| Convergence angle | 35.9° |

### Hillshade Comparison

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
![SETSM 2m](../examples/figures/ucsd-setsm2m-hillshade.png)
**SETSM 2m DEM**
:::
::::

The boundary-flag approach runs cleanly with no tile-boundary artifacts, but the resulting DEM is still noise-dominated — no buildings, streets, or other urban structure are visible. Under the same input data and ~20 min of processing, ASP resolves individual buildings and block-level topography. SETSM was developed for and is most often applied to polar/mountain/glacier scenes; default parameters on a mid-latitude urban WorldView-3 pair (convergence 35.9°) do not appear to produce a usable DSM out of the box.

:::{dropdown} Additional notes
:icon: note

- SETSM's user manual warns against `-seed <filepath> <sigma>`: *"We caution that it is better to not use a seed DEM if possible, as it can only negatively impact the quality of the SETSM DEM."*
- The run did not produce `results_hillshade.tif`; the hillshade above was generated from `results_dem.tif` with `gdaldem hillshade`.
:::

## References

- [SETSM repository](https://github.com/setsmdeveloper/SETSM)
- [SETSM user manual](https://github.com/setsmdeveloper/SETSM/blob/master/SETSM_User_manual.pdf)
- Noh & Howat (2015), "Automated stereo-photogrammetric DEM generation at high latitudes: Surface Extraction with TIN-based Search-space Minimization (SETSM) validation and demonstration over glaciated regions", *GIScience & Remote Sensing*, 52(2), 198-217.
