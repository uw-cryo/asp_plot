# SETSM vs ASP — SpaceNet UCSD WorldView-3

Compare DEMs produced by two open-source stereo photogrammetry pipelines on the same WorldView-3 stereo pair:

- **ASP** (NASA Ames Stereo Pipeline) — processed in the [UCSD notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo)
- **[SETSM](https://github.com/setsmdeveloper/SETSM)** (Surface Extraction with TIN-based Search-space Minimization) — Ohio State / Polar Geospatial Center

Both pipelines accept raw satellite imagery with RPC camera models and produce gridded DSMs.

## Source Data

IARPA CORE3D SpaceNet UCSD dataset — two WorldView-3 panchromatic acquisitions:

| Catalog ID | Date | Off-nadir | Image size |
|---|---|---|---|
| 1040010007A93700 | 2015-02-12 | 8.4° | 43008 × 46080 px |
| 1040010007CA4D00 | 2015-02-24 | 12.9° | 43008 × 46080 px |

This is the same pair (`21deg_12d`, convergence 21.2°) used in the [scene-selection notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo_scene_selection) and the [ASP processing notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo).

## Approach

SETSM is open-source (Apache 2.0), pure C++ with no GPU requirement. We built and ran it via Docker to avoid host dependency conflicts.

SETSM reads only `*.tif` and `*.raw` images, so the source NTFs must be converted up-front. Section 2.1.4 of the [SETSM User Manual](https://github.com/setsmdeveloper/SETSM/blob/master/SETSM_User_manual.pdf) provides the exact `gdal_translate` command:

```bash
gdal_translate -q --config GDAL_CACHEMAX 2048 -ot UInt16 -co NBITS=16 \
    -co bigtiff=if_safer -co tiled=yes -co compress=lzw input.ntf output.tif
```

The DigitalGlobe NTFs use JPEG2000 internal compression, so the GDAL build doing the conversion needs a JP2 driver — ASP's bundled `gdal_translate` (e.g. `~/asp/dev/bin/gdal_translate`) ships with `JP2OpenJPEG`. The output TIFFs end up larger than the source NTFs (~2.4 GB each here) because LZW does not compress 11-bit panchromatic content as efficiently as JPEG2000.

SETSM then works on those TIFFs with RPC camera models read from the DigitalGlobe XML metadata files (the XMLs live alongside the TIFFs and share the basename). Memory is the limiting factor on a laptop:

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
    -image 1040010007A93700_P001.tif -image 1040010007CA4D00_P001.tif \
    -outpath /data/results -outres 2 -mem 10 -minH 0 -maxH 300 \
    -boundary_min_X 476000 -boundary_min_Y 3635600 \
    -boundary_max_X 479000 -boundary_max_Y 3638600 \
    -projection utm
```

The boundary defines a 3 km × 3 km UTM 11N area matching the ASP processing extent.

| Metric | Value |
|---|---|
| Computation time | ~20 min |
| Peak memory (SETSM-reported) | 2.72 GB |
| Docker `--memory` limit | 11 GB |
| Output DEM | 1500 × 1500 px, 2 m, EPSG:32611 |
| Output std dev | 39.6 m |
| Convergence angle | 21.2° |

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

SETSM resolves the same urban structure as ASP — buildings, streets, and the campus / Mount Soledad topography are all clearly visible. The SETSM hillshade is somewhat speckled compared to ASP's, but the underlying geometry is recovered. No tile-boundary artifacts.

:::{dropdown} Additional notes
:icon: note

- SETSM's user manual warns against `-seed <filepath> <sigma>`: *"We caution that it is better to not use a seed DEM if possible, as it can only negatively impact the quality of the SETSM DEM."*
- The run did not produce `results_hillshade.tif`; the hillshade above was generated from `results_dem.tif` with a matplotlib `LightSource` shader.
:::

## References

- [SETSM repository](https://github.com/setsmdeveloper/SETSM)
- [SETSM user manual](https://github.com/setsmdeveloper/SETSM/blob/master/SETSM_User_manual.pdf)
- Noh & Howat (2015), "Automated stereo-photogrammetric DEM generation at high latitudes: Surface Extraction with TIN-based Search-space Minimization (SETSM) validation and demonstration over glaciated regions", *GIScience & Remote Sensing*, 52(2), 198-217.
