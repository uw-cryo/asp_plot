# bundle_adjust_cameras

The `bundle_adjust_cameras` command-line tool shows what `bundle_adjust` did to the cameras: how far each camera center moved, how each camera rotated, and — when ASP wrote it — how far the triangulated points moved on the ground. It reads only the files `bundle_adjust` writes into its own output folder (the per-camera `*.adjust` translation + rotation, `*camera_offsets.txt`, and `*triangulation_offsets.txt`, with each camera's absolute position from its `*.adjusted_state.json` or, for DigitalGlobe runs without one, from the original `.xml` ephemeris), so **no original camera files are needed**. This is the same figure `asp_report` adds after the bundle adjustment residual pages when `--bundle-adjust-prefix` is given.

```{figure} ../figures/example_bundle_adjust_cameras.png
:alt: bundle_adjust_cameras summary figure for a five-scene WorldView-2 multi-view run over Atlanta
:width: 100%

Five-scene WorldView-2 multi-view run over Atlanta (`notebooks/WorldView/worldview_spacenet_atlanta_mvs.ipynb`). Top: horizontal and vertical change of each camera center in meters. Middle: roll / pitch / yaw change in degrees, the value printed on every bar; the satellite cartoon is a legend for the body axes and is not to scale. Bottom (ASP >= 3.6 only): median and mean change of each image's triangulated points. Here the cameras moved 24–70 m but the ground points moved under 1 m — the solver traded camera position against orientation.
```

## Basic usage

Point the tool at the `bundle_adjust` output folder. The figure is saved into that folder as `bundle_adjust_cameras_summary.png`:

```bash
bundle_adjust_cameras --directory path/to/ba/
```

## Map projection and output location

`--map-crs` sets the CRS of the camera-center geometry only (the east/north/up offsets do not depend on it). `--output-directory` and `--output-filename` choose where the figure goes:

```bash
bundle_adjust_cameras --directory path/to/ba/ \
                      --map-crs EPSG:32616 \
                      --title "Atlanta 5-scene MVS" \
                      --output-directory path/to/figures/
```

## DigitalGlobe runs

A `bundle_adjust` run on DigitalGlobe cameras (`-t dg`) writes only the `.adjust` deltas, without `*.adjusted_state.json`. The tool then takes each camera's position from the original `.xml` ephemeris, which it looks for in the `bundle_adjust` folder and its parent. Point it elsewhere if the XMLs live somewhere else:

```bash
bundle_adjust_cameras --directory path/to/ba/ \
                      --original-cameras-directory path/to/xml_cameras/
```

## Reading the figure

- **Camera-center change** comes from `camera_offsets.txt` when ASP wrote it (the title says which); otherwise from the `.adjust` translation. Values are meters, at the camera's upper-left pixel, and — per ASP — measured *after* any `--initial-transform`.
- **Orientation change** is the `.adjust` rotation as roll (about the along-track X axis), pitch (about across-track Y), and yaw (about nadir Z), in degrees. The cartoon is a legend only.
- **Triangulated-point change** (ASP >= 3.6) is the median and mean distance between each image's initial and final triangulated points, with the point count. A large camera shift next to a small ground change means the solver traded position against orientation; the ground row is what matters for the DEM.
- A run that applied only an identity transform (the recipe for recovering the unadjusted cameras, see the [UCSD notebook](../examples/notebooks/worldview_spacenet_ucsd_stereo.ipynb)) shows a "no camera change" note.

To go further — how the position and orientation change *along the trajectory* of a linescan camera — use [`csm_camera_plot`](csm_camera_plot.md), which compares the original and optimized CSM camera models.

## Full options

```
Usage: bundle_adjust_cameras [OPTIONS]

  Visualize before/after camera positions from a bundle_adjust folder.

  Reads the self-contained camera products written by bundle_adjust
  (``*.adjust``, ``*.adjusted_state.json``, and, when present,
  ``*camera_offsets.txt``) and produces a summary: per-camera bars of the
  horizontal and vertical camera-center change, above per-camera bars of the
  roll/pitch/yaw orientation change with the degrees printed on each bar and
  one satellite cartoon as the legend for the body axes, and -- when the run
  wrote ``triangulation_offsets.txt`` (ASP >= 3.6) -- a third row of the per-
  image triangulated-point change, the ground effect of the camera change.

  Unlike ``csm_camera_plot``, this does not require the pre-adjustment
  original camera files -- it works directly on the bundle_adjust output.

Options:
  --directory TEXT                Path to the bundle_adjust output directory
                                  (the folder holding the *.adjust,
                                  *.adjusted_state.json, and optional
                                  *camera_offsets.txt files). No default. Must
                                  be supplied.
  --map-crs TEXT                  CRS for the camera-center geometry, as
                                  EPSG:XXXX (e.g. the site's UTM zone). Only
                                  affects the returned geometry; the
                                  east/north/up offsets do not depend on it.
                                  Default: geographic coordinates (EPSG:4326).
  --original-cameras-directory TEXT
                                  Directory holding the original .xml cameras,
                                  used only for DigitalGlobe runs that lack
                                  *.adjusted_state.json. If not supplied, the
                                  bundle_adjust directory and its parent are
                                  searched automatically.
  --title TEXT                    Optional title for the summary figure.
                                  Default: None.
  --output-directory TEXT         Directory to save the figure. Default: the
                                  bundle_adjust directory itself.
  --output-filename TEXT          Figure filename. Default:
                                  bundle_adjust_cameras_summary.png.
  --help                          Show this message and exit.
```
