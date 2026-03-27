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

## Next Steps

### 1. Build and run SETSM via Docker

SETSM is open-source (Apache 2.0), pure C++ with no GPU requirement. Building in Docker avoids any risk of dependency conflicts with the host system (conda, Homebrew, PROJ version mismatches, etc.).

Repository: https://github.com/setsmdeveloper/SETSM

#### Build the Docker image

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

ENTRYPOINT ["setsm"]
DOCKERFILE
```

This uses a multi-stage build — the final image is small and only contains the `setsm` binary and its runtime libraries.

#### Run SETSM on the cropped images

Mount the data directory into the container and run:

```bash
docker run --platform linux/amd64 --rm \
    -v /path/to/ucsd_stereo_SETSM:/data \
    -w /data \
    setsm \
    -image 1040010007A3D100_P001.tif \
    -image 1040010007A93700_P001.tif \
    -outpath /data/results -outres 2 -mem 16
```

Optional flags:
- `-minH 0 -maxH 300` — constrain terrain height range (UCSD campus is ~0–170 m) to speed processing
- `-tilesize 100000` — disable tiling (appropriate for this small crop)

Expected outputs in `results/`:
- `*_dem.tif` — DSM (float32 GeoTIFF)
- `*_ortho.tif` — orthorectified image (16-bit GeoTIFF)
- `*_matchtag.tif` — binary match mask (1 = matched, 0 = interpolated)
- `*_meta.txt` — metadata file

**Note:** Running `linux/amd64` via Rosetta on Apple Silicon works but is slower than native. Processing the cropped ~10k x 12k px images at 2 m resolution should still be tractable.

### 3. Compare ASP and SETSM DEMs

Once both DEMs exist for the same area:

1. **Difference map** — Use `asp_plot`'s raster differencing to compute SETSM DEM minus ASP DEM
2. **Hillshade comparison** — Visual comparison of surface detail and artifacts
3. **ICESat-2 validation** — Compare both DEMs against ICESat-2 ATL06-SR altimetry (already available from the ASP notebook) to get independent accuracy metrics (median, NMAD)
4. **Match coverage** — Compare SETSM's `matchtag` with ASP's intersection error / triangulation maps to assess where each pipeline produces valid matches

### 4. Potential issues to watch for

- **CRS alignment** — ASP and SETSM may output DEMs in different projections. Reproject to a common CRS before differencing.
- **Datum** — Both should use WGS84 ellipsoidal heights, but verify.
- **Resolution** — Match output GSD for fair comparison (both at 2 m).
- **No-data handling** — Each pipeline has different conventions for no-data / void fill regions.

## References

- SETSM repository: https://github.com/setsmdeveloper/SETSM
- SETSM user manual: https://github.com/setsmdeveloper/SETSM/blob/master/SETSM_User_manual.pdf
- Noh & Howat (2015), "Automated stereo-photogrammetric DEM generation at high latitudes: Surface Extraction with TIN-based Search-space Minimization (SETSM) validation and demonstration over glaciated regions", *GIScience & Remote Sensing*, 52(2), 198-217.
- ASP documentation: https://stereopipeline.readthedocs.io
- SpaceNet UCSD dataset: https://spacenet.ai/core3d/
