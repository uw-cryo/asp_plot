# stereo_geom

The `stereo_geom` command-line tool creates visualizations of stereo geometry for satellite imagery from its camera metadata. It produces a combined plot with a skyplot showing satellite viewing angles and a map view showing the footprints and satellite positions.

The sensor is detected from the files themselves, so the same command works across WorldView (and other DigitalGlobe-heritage) XML, the Airbus DIMAP families, ASTER `gen_aster` XML, and RPC-only products.

## Basic usage

Pass the camera metadata files directly. `INPUTS` may be any mix of files, directories, and glob patterns, and need not follow a fixed directory structure:

```bash
# A shell glob expands to the candidate XML files
stereo_geom *.XML

# Explicit files
stereo_geom scene1.xml scene2.xml

# A delivery directory (searched recursively)
stereo_geom my_delivery_dir/
```

Directories are searched recursively, and non-camera files (e.g. `README.XML`, ortho `*_ortho.xml`) are skipped automatically. By default, the tool saves the output as `<directory_name>_stereo_geom.png` in the common input directory.

### RPC-only products

Cartosat-1, Deimos, and other products ASP runs with `-t rpc` have no camera XML — the rational polynomial coefficients live in the image itself, or in a `*_RPC.TXT` sidecar. Point the tool at the **images**:

```bash
stereo_geom fore.tif aft.tif

# ...or at the directory holding them
stereo_geom my_rpc_delivery/
```

Give it the raw delivered images, not map-projected ones: an RPC describes the original image grid, so an orthorectified raster is skipped even if it kept its RPC metadata. The geometry is derived from the camera model rather than parsed, so these products report a footprint, satellite azimuth/elevation, off-nadir angle and GSD, but no attitude or sun angles, and no acquisition time unless the image header records one.

## More than two scenes (multi-view)

`stereo_geom` is not limited to a stereo pair. Give it any number of scenes and
it assesses the geometry of every pair:

```bash
stereo_geom scene1.xml scene2.xml scene3.xml scene4.xml
```

With more than two scenes the tool writes, into the output directory:

- **one overview figure** with all scenes color-coded (skyplot of every satellite
  position + a map of every footprint and ground track) →
  `<name>_stereo_geom_overview.png`, and
- **one figure per pair** — every combination of two scenes — each with the full
  pairwise stereo stats (convergence angle, B:H ratio, BIE, asymmetry,
  intersection area) in its title → `<name>_stereo_geom_<catidA>_<catidB>.png`.

So four scenes produce one overview plus six pair figures. Pairs whose footprints
do not overlap are still plotted (their intersection-dependent stats show `N/A`).
With exactly two scenes the output is a single `<name>_stereo_geom.png`, as
before.

## Using `--directory` instead

When no positional `INPUTS` are given, the tool falls back to `--directory` (default: current directory). This is the original interface and remains supported:

```bash
stereo_geom --directory /path/to/directory/with/xml/files
```

## Custom output location

```bash
stereo_geom scene1.xml scene2.xml \
            --output-directory /path/to/save/plots \
            --output-filename custom_output.png
```

## Without basemap

A basemap is added to the map view by default, which requires an internet connection. To run offline, skip it:

```bash
stereo_geom my_delivery_dir/ \
            --no-basemap
```

## Full options

```
Usage: stereo_geom [OPTIONS] [INPUTS]...

  Generate stereo geometry plots from satellite camera metadata files.

  The sensor is detected from the files themselves: WorldView (and other
  DigitalGlobe-heritage) XML, Airbus DIMAP v2 (Pléiades 1A/1B and Neo, SPOT
  6/7, PeruSat-1), DIMAP v1 (SPOT 5, ALOS PRISM), ASTER (ASP gen_aster camera
  XML), and RPC-only products such as Cartosat-1 and Deimos, whose camera
  model lives in the image itself.

  This tool creates a skyplot and map visualization of the satellite positions
  and ground footprints. INPUTS may be any mix of camera metadata files,
  images carrying RPCs, directories, and glob patterns, and need not follow a
  fixed directory structure, e.g.:

      stereo_geom *.XML

      stereo_geom scene1.xml scene2.xml

      stereo_geom fore.tif aft.tif

      stereo_geom my_delivery_dir/

  If no INPUTS are given, --directory is used (default: current directory).

Options:
  --directory TEXT         Directory containing camera metadata files (XMLs,
                           or images carrying RPCs) for stereo geometry
                           analysis. Used when no positional INPUTS are given.
                           Default: current directory.
  --no-basemap             Skip the figure basemaps (basemaps are added by
                           default, which requires an internet connection).
  --output-directory TEXT  Directory to save the output plot. Default: Input
                           directory.
  --output-filename TEXT   Filename for the output plot. Default: Directory
                           name with _stereo_geom.png suffix. With more than
                           two scenes this is the stem for the per-pair and
                           overview figures.
  --help                   Show this message and exit.
```
