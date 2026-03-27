# SETSM vs ASP DEM Comparison — SpaceNet UCSD WorldView-3

## Goal

Compare DEMs produced by two open-source stereo photogrammetry pipelines on the same WorldView-3 stereo pair:

- **ASP** (NASA Ames Stereo Pipeline) — already processed in `worldview_spacenet_ucsd_stereo.ipynb`
- **SETSM** (Surface Extraction with TIN-based Search-space Minimization) — to be processed

Both pipelines accept raw satellite imagery with RPC camera models and produce gridded DSMs.

## Source Data

IARPA CORE3D SpaceNet UCSD dataset — two WorldView-3 panchromatic acquisitions:

| Catalog ID | Date | Off-nadir | Image size |
|---|---|---|---|
| 1040010007A3D100 | 2015-02-11 | 24.3° | 43008 x 44032 px |
| 1040010007A93700 | 2015-02-12 | 9.6° | 43008 x 46080 px |

Full NTF images (~800 MB each) are in `ucsd_stereo/images/`. The ASP notebook processes a cropped ~3 km region around the UCSD campus.

## Cropping Raw Images for SETSM

SETSM works on raw (non-mapprojected) imagery with RPC camera models read from DigitalGlobe XML metadata files. The full images are too large for a quick test, so we cropped to the same region as the ASP processing.

### Why cropping preserves RPCs

The RPC model maps ground coordinates to image coordinates:

```
line = f(lat, lon, height) * LINE_SCALE + LINE_OFFSET
sample = g(lat, lon, height) * SAMP_SCALE + SAMP_OFFSET
```

When cropping at pixel offset `(col_off, row_off)`, only the offsets change:

- `new LINE_OFFSET = old LINE_OFFSET - row_off`
- `new SAMP_OFFSET = old SAMP_OFFSET - col_off`

All polynomial coefficients and scale factors remain identical. The RPC model is still mathematically valid.

### Crop procedure

1. **Define geographic extent** — ASP's mapprojected crop area (UTM 11N: 476000–479000 E, 3635600–3638600 N) plus a 500 m buffer on each side for SETSM context.

2. **Compute pixel windows** — Used GDAL's RPC transformer to convert the geographic extent to pixel coordinates in each image (different per image due to different viewing geometries), plus 50 px padding.

   | Image | Crop srcwin (col, row, w, h) | Cropped size |
   |---|---|---|
   | 1040010007A3D100 | 11189, 16662, 10575, 12163 | 10575 x 12163 px |
   | 1040010007A93700 | 9318, 17173, 12789, 12927 | 12789 x 12927 px |

3. **Crop NTF to GeoTIFF** — Used ASP's `gdal_translate` (which has the JP2OpenJPEG driver needed for JPEG2000-compressed NTF files):

   ```bash
   gdal_translate -of GTiff -srcwin 11189 16662 10575 12163 \
       ucsd_stereo/1040010007A3D100_P001.NTF \
       ucsd_stereo_SETSM/1040010007A3D100_P001.tif

   gdal_translate -of GTiff -srcwin 9318 17173 12789 12927 \
       ucsd_stereo/1040010007A93700_P001.NTF \
       ucsd_stereo_SETSM/1040010007A93700_P001.tif
   ```

   `gdal_translate` automatically adjusts the RPC offsets in the output GeoTIFF metadata.

4. **Create modified XML files** — SETSM reads RPCs from DigitalGlobe XML files (not from GeoTIFF metadata). Copied the original XMLs and updated:
   - `LINEOFFSET` and `SAMPOFFSET` (subtracted the row/col crop offsets)
   - `NUMROWS` and `NUMCOLUMNS` (set to cropped dimensions)
   - `TIL` tile corner offsets (reset to match cropped dimensions)

5. **Verification** — Confirmed that:
   - XML and GeoTIFF RPC offsets match exactly
   - Center pixels in the cropped images map to identical ground coordinates as the corresponding pixels in the original images (zero error)

### Output files

Located in `ucsd_stereo_SETSM/`:

```
1040010007A3D100_P001.tif   (169 MB, 10575 x 12163 px)
1040010007A3D100_P001.xml   (modified DigitalGlobe XML with adjusted RPCs)
1040010007A93700_P001.tif   (217 MB, 12789 x 12927 px)
1040010007A93700_P001.xml   (modified DigitalGlobe XML with adjusted RPCs)
```

## Building and Running SETSM via Docker

SETSM is open-source (Apache 2.0), pure C++ with no GPU requirement. Building in Docker avoids any risk of dependency conflicts with the host system (conda, Homebrew, PROJ version mismatches, etc.).

Repository: https://github.com/setsmdeveloper/SETSM

### Build the Docker image

```bash
# Clone SETSM
git clone https://github.com/setsmdeveloper/SETSM.git
cd SETSM

# Build the image (uses linux/amd64 via Rosetta on Apple Silicon)
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

This uses a multi-stage build — the final image is small and only contains the `setsm` binary and its runtime libraries. The `default.txt` configuration file is bundled into the image and copied into the working directory at runtime (SETSM requires it to be present alongside the data).

### Run SETSM on the cropped images

Mount the data directory into the container and run:

```bash
docker run --platform linux/amd64 --rm \
    -v /path/to/ucsd_stereo_SETSM:/data \
    -w /data \
    setsm \
    -image 1040010007A3D100_P001.tif \
    -image 1040010007A93700_P001.tif \
    -outpath /data/results -outres 2 -mem 16 \
    -minH 0 -maxH 300
```

- `-outres 2` — 2 m output resolution (matches ASP processing)
- `-mem 16` — limit memory usage to 16 GB
- `-minH 0 -maxH 300` — constrain terrain height search range (UCSD campus is ~0–170 m) to speed processing

Expected outputs in `results/`:
- `*_dem.tif` — DSM (float32 GeoTIFF)
- `*_ortho.tif` — orthorectified image (16-bit GeoTIFF)
- `*_matchtag.tif` — binary match mask (1 = matched, 0 = interpolated)
- `*_meta.txt` — metadata file

### Run 1: Cropped images (ucsd_stereo_SETSM_crop)

SETSM v4.3.16 on the cropped images. Run on an Apple Silicon MacBook Air (M2, 8 GB) via Docker with `linux/amd64` emulation (Rosetta).

```bash
docker run --platform linux/amd64 --rm \
    -v /path/to/ucsd_stereo_SETSM_crop:/data -w /data setsm \
    -image 1040010007A3D100_P001.tif -image 1040010007A93700_P001.tif \
    -outpath /data/results -outres 2 -mem 16 -minH 0 -maxH 300
```

| Metric | Value |
|---|---|
| Total computation time | ~115 minutes |
| Peak memory usage | 3.96 GB |
| Tiling | 4x4 grid (16 tiles, ~1 km each) |
| Output DEM size | 1986 x 2019 px (2 m resolution) |
| Output CRS | WGS 84 / UTM zone 11N (EPSG:32611) |
| Elevation range | -99.5 to 540.6 m (mean 118.3 m) |
| Convergence angle | 35.9° |
| Expected height accuracy | 1.09 m |

**Result: Poor quality.** The hillshade showed unreasonable noise and elevation blunders throughout, rendering quality analysis unnecessary.

**Notes:**
- Ortho TIFFs were written with corrupt LZW compression (unreadable in QGIS/GDAL). Likely a libtiff version issue inside the Docker container. The DEM and matchtag were fine.
- The `-minH 0 -maxH 300` range was slightly exceeded in the output; SETSM uses these as search bounds, not hard clamps.
- The SETSM user manual's `-seed <filepath> <sigma>` flag was investigated but is documented as a performance optimization only, not a quality improvement: *"We caution that it is better to not use a seed DEM if possible, as it can only negatively impact the quality of the SETSM DEM."*

### Run 2: Full scene — OOM failure

Attempted to run on the full uncropped images (43k×44k and 43k×46k px) to rule out cropping as the cause of poor results. SETSM loads both full images into memory regardless of output resolution — two UInt16 images plus pyramids required well over 14 GB. The container was OOM-killed (exit code 137) during preprocessing at both `--memory 10g` and `--memory 14g` on a 16 GB machine. Full-scene SETSM processing is not feasible locally.

### Troubleshooting

- **`'default.txt' file doesn't exist`** — SETSM requires a `default.txt` config file in the working directory. The Dockerfile above handles this automatically. If running a manually-built binary, copy `default.txt` from the SETSM repo into the data directory.
- **JPEG2000 NTF read errors** — The conda `asp_plot` environment lacks a JP2 driver. Use ASP's `gdal_translate` for cropping NTF files (it includes the JP2OpenJPEG driver).

## Conclusion

The cropped-image run produced a DEM with excessive noise and blunders, and the full-scene run could not complete on a 16 GB laptop due to SETSM's memory requirements. A proper comparison of ASP vs SETSM DEM quality would require processing on a machine with more RAM (the SETSM manual recommends nodes with 12+ cores and proportionally more memory for full-resolution commercial imagery).

## References

- SETSM repository: https://github.com/setsmdeveloper/SETSM
- SETSM user manual: https://github.com/setsmdeveloper/SETSM/blob/master/SETSM_User_manual.pdf
- Noh & Howat (2015), "Automated stereo-photogrammetric DEM generation at high latitudes: Surface Extraction with TIN-based Search-space Minimization (SETSM) validation and demonstration over glaciated regions", *GIScience & Remote Sensing*, 52(2), 198-217.
- ASP documentation: https://stereopipeline.readthedocs.io
- SpaceNet UCSD dataset: https://spacenet.ai/core3d/
