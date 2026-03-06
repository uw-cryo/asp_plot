# stereo_geom

The `stereo_geom` command-line tool creates visualizations of stereo geometry for satellite imagery based on the XML camera files. It produces a combined plot with a skyplot showing satellite viewing angles and a map view showing the footprints and satellite positions.

## Basic usage

```bash
stereo_geom --directory /path/to/directory/with/xml/files
```

By default, the tool saves the output as `<directory_name>_stereo_geom.png` in the input directory.

## Custom output location

```bash
stereo_geom --directory /path/to/directory/with/xml/files \
            --output_directory /path/to/save/plots \
            --output_filename custom_output.png
```

## With basemap

Adding a basemap to the map view requires an internet connection:

```bash
stereo_geom --directory /path/to/directory/with/xml/files \
            --add_basemap True
```

## Full options

```
Usage: stereo_geom [OPTIONS]

  Generate stereo geometry plots for DigitalGlobe/Maxar XML files. This tool
  creates a skyplot and map visualization of the satellite positions and
  ground footprints.

Options:
  --directory TEXT         Directory containing XML files for stereo geometry
                           analysis. Default: current directory.
  --add_basemap BOOLEAN    If True, add a basemap to the figures, which
                           requires internet connection. Default: True.
  --output_directory TEXT  Directory to save the output plot. Default: Input
                           directory.
  --output_filename TEXT   Filename for the output plot. Default: Directory
                           name with _stereo_geom.png suffix.
  --help                   Show this message and exit.
```
