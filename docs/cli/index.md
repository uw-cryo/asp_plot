# CLI Tools

`asp_plot` provides seven command-line tools for different visualization tasks:

::::{grid} 1
:gutter: 3

:::{grid-item-card} asp_report
:link: asp_report
:link-type: doc

Generate comprehensive PDF reports of ASP stereo processing results.
:::

:::{grid-item-card} stereo_geom
:link: stereo_geom
:link-type: doc

Visualize stereo acquisition geometry from satellite camera metadata.
:::

:::{grid-item-card} csm_camera_plot
:link: csm_camera_plot
:link-type: doc

Diagnostic plots for CSM camera model adjustments after bundle adjustment or jitter correction.
:::

:::{grid-item-card} bundle_adjust_cameras
:link: bundle_adjust_cameras
:link-type: doc

How far each camera moved and rotated in a `bundle_adjust` run, from the run's own output folder — no original cameras needed.
:::

:::{grid-item-card} request_planetary_altimetry
:link: request_planetary_altimetry
:link-type: doc

Submit LOLA (Moon) or MOLA (Mars) altimetry data requests for planetary DEM validation.
:::

:::{grid-item-card} gallery
:link: gallery
:link-type: doc

Lay out many DEMs as a grid of thumbnails sharing one color scale, for QA'ing a stack of ASP outputs at a glance.
:::

:::{grid-item-card} dem_benchmark
:link: dem_benchmark
:link-type: doc

Score many DEMs — scene combinations, MVS vs. pairwise + `dem_mosaic`, parameter sweeps — against one ICESat-2 or LOLA/MOLA sample, side by side.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

asp_report
stereo_geom
csm_camera_plot
bundle_adjust_cameras
request_planetary_altimetry
gallery
dem_benchmark
```
