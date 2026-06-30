# stereo_geom

The `stereo_geom` command-line tool creates visualizations of stereo geometry for satellite imagery based on the XML camera files. It produces a combined plot with a skyplot showing satellite viewing angles and a map view showing the footprints and satellite positions.

## Basic usage

Pass the XML camera files directly. `INPUTS` may be any mix of XML files, directories, and glob patterns, and need not follow a fixed directory structure:

```bash
# A shell glob expands to the candidate XML files
stereo_geom *.XML

# Explicit files
stereo_geom scene1.xml scene2.xml

# A delivery directory (searched recursively for camera XMLs)
stereo_geom my_delivery_dir/
```

Directories are searched recursively for camera XMLs, and non-camera files (e.g. `README.XML`, ortho `*_ortho.xml`) are skipped automatically. By default, the tool saves the output as `<directory_name>_stereo_geom.png` in the common input directory.

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
            --output_directory /path/to/save/plots \
            --output_filename custom_output.png
```

## With basemap

Adding a basemap to the map view requires an internet connection:

```bash
stereo_geom my_delivery_dir/ \
            --add_basemap True
```

## Full options

```
Usage: stereo_geom [OPTIONS] [INPUTS]...

  Generate stereo geometry plots for WorldView XML files.

  This tool creates a skyplot and map visualization of the satellite positions
  and ground footprints. INPUTS may be any mix of XML files, directories, and
  glob patterns and need not follow a fixed directory structure, e.g.:

      stereo_geom *.XML

      stereo_geom scene1.xml scene2.xml

      stereo_geom my_delivery_dir/

  If no INPUTS are given, --directory is used (default: current directory).

Options:
  --directory TEXT         Directory containing XML files for stereo geometry
                           analysis. Used when no positional INPUTS are given.
                           Default: current directory.
  --add_basemap BOOLEAN    If True, add a basemap to the figures, which
                           requires internet connection. Default: True.
  --output_directory TEXT  Directory to save the output plot. Default: Input
                           directory.
  --output_filename TEXT   Filename for the output plot. Default: Directory
                           name with _stereo_geom.png suffix.
  --help                   Show this message and exit.
```
