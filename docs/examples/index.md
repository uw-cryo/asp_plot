# Example Notebooks

Examples of modular usage of `asp_plot`, organized by sensor type. Each notebook demonstrates the plotting classes and functions available for different satellite instruments.

## Earth-based

::::{grid} 1
:gutter: 3

:::{grid-item-card} WorldView — SpaceNet Atlanta (Multi-View Stereo)
:link: notebooks/worldview_spacenet_atlanta_mvs
:link-type: doc

Three-scene same-pass multi-view stereo of publicly available SpaceNet Atlanta WorldView-2 data, compared against the three pairwise runs merged with `dem_mosaic`, and scored with `dem_benchmark` against single pairs and a five-scene run on one ICESat-2 sample.
:::

:::{grid-item-card} WorldView — SpaceNet Atlanta (Scene-Combination Benchmark)
:link: notebooks/worldview_spacenet_benchmark
:link-type: doc

Which combination of N scenes and which processing flow give the best DEM: all ten pairs, six `dem_mosaic` blends and six multi-view runs of five same-pass Atlanta scenes, then the same experiment at UCSD with multi-date WorldView-3 over steep urban terrain — 38 DEMs scored with `dem_benchmark` on one ICESat-2 sample per site.
:::

:::{grid-item-card} WorldView — SpaceNet Atlanta (Full-Scene Benchmark)
:link: notebooks/worldview_spacenet_atlanta_full_benchmark
:link-type: doc

The scene-combination benchmark repeated on the full strips of the five Atlanta scenes: 21 DEMs from ten pairs (mapprojected and raw), two multi-view runs and three blends, scored on stable terrain. Tests the cropped study's rules at strip scale, compares raw with mapprojected stereo, and shows along-track camera errors that a crop averages out.
:::

:::{grid-item-card} WorldView — SpaceNet UCSD (Full-Scene Benchmark)
:link: notebooks/worldview_spacenet_ucsd_full_benchmark
:link-type: doc

The scene-combination benchmark's second site on the full scenes: five multi-date WorldView-3 collects over steep urban terrain, 18 DEMs from ten pairs (mapprojected, three also raw), a multi-view run and four blends, scored on stable terrain. Tests whether the crop's UCSD findings survive a 17× larger sample, compares the blends, and asks whether mapprojection helps on slopes.
:::

:::{grid-item-card} WorldView — Full-Scene DEMs vs. 3DEP Lidar (Atlanta and UCSD)
:link: notebooks/worldview_spacenet_lidar_comparison
:link-type: doc

The 39 full-scene benchmark DEMs of both sites scored against USGS 3DEP 1 m lidar on open ground: about 11 M shared cells per site instead of 14,000–38,000 ICESat-2 points. Checks the ICESat-2 rankings, scores the steep ground ICESat-2 barely samples, and shows along-track ripples at Atlanta.
:::

:::{grid-item-card} WorldView — SpaceNet Atlanta (Jitter Correction)
:link: notebooks/worldview_spacenet_atlanta_jitter
:link-type: doc

ASP `jitter_solve` on the five same-pass Atlanta WorldView-2 strips, with every pair re-triangulated from its archived disparity. Measures how much attitude jitter costs narrow and wide pairs, and whether correcting it changes the benchmark's blends and the separability of its DEMs.
:::

:::{grid-item-card} WorldView — SpaceNet Atlanta (Scene Selection)
:link: notebooks/worldview_spacenet_atlanta_stereo_scene_selection
:link-type: doc

Pair-ranking and DEM-vs-ICESat-2 comparison used to choose the Atlanta scenes processed in the companion multi-view notebook.
:::

:::{grid-item-card} WorldView — SpaceNet UCSD
:link: notebooks/worldview_spacenet_ucsd_stereo
:link-type: doc

Stereo processing of publicly available IARPA CORE3D UCSD WorldView-3 data.
:::

:::{grid-item-card} WorldView — SpaceNet UCSD (Scene Selection)
:link: notebooks/worldview_spacenet_ucsd_stereo_scene_selection
:link-type: doc

Pair-ranking and DEM-vs-ICESat-2 comparison used to choose the UCSD stereo pair processed in the companion notebook.
:::

:::{grid-item-card} WorldView — Uyuni Jitter Plots
:link: notebooks/worldview_uyuni_jitter_plots
:link-type: doc

CSM camera model comparison plots after jitter correction for WorldView imagery.
:::

:::{grid-item-card} Pléiades Neo — Marseille Tri-Stereo
:link: notebooks/pleiades_neo_marseille_tristereo
:link-type: doc

Tri-stereo processing of the free Airbus Pléiades Neo sample over Marseille: DIMAP stereo geometry analysis, bundle adjustment, and a three-scene multi-view stereo DEM.
:::

:::{grid-item-card} ASTER — With Map-projection
:link: notebooks/aster_with_mapprojection
:link-type: doc

ASTER stereo processing with map-projected imagery.
:::

:::{grid-item-card} ASTER — With Bundle Adjust and Jitter Correction
:link: notebooks/aster_with_bundle_adjust_and_jitter_correction
:link-type: doc

ASTER processing with bundle adjustment and jitter correction.
:::

::::

## Planetary

::::{grid} 1
:gutter: 3

:::{grid-item-card} Lunar Reconnaissance Orbiter NAC
:link: notebooks/lunar_recon_orbiter
:link-type: doc

LRO Narrow Angle Camera stereo processing on the lunar surface.
:::

:::{grid-item-card} Mars MGS MOC NA
:link: notebooks/mars_mgs_orbital_camera
:link-type: doc

Mars Global Surveyor MOC Narrow Angle stereo, both mapprojected and non-mapprojected, with MOLA `pc_align`.
:::

:::{grid-item-card} Mars MRO CTX
:link: notebooks/mars_mro_ctx
:link-type: doc

Mars Reconnaissance Orbiter Context Camera processing.
:::

:::{grid-item-card} Mars MRO HiRISE
:link: notebooks/mars_mro_hirise
:link-type: doc

Mars Reconnaissance Orbiter High Resolution Imaging Science Experiment processing.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

notebooks/worldview_spacenet_atlanta_mvs
notebooks/worldview_spacenet_benchmark
notebooks/worldview_spacenet_atlanta_full_benchmark
notebooks/worldview_spacenet_ucsd_full_benchmark
notebooks/worldview_spacenet_lidar_comparison
notebooks/worldview_spacenet_atlanta_jitter
notebooks/worldview_spacenet_atlanta_stereo_scene_selection
notebooks/worldview_spacenet_ucsd_stereo
notebooks/worldview_spacenet_ucsd_stereo_scene_selection
notebooks/worldview_uyuni_jitter_plots
notebooks/pleiades_neo_marseille_tristereo
notebooks/aster_with_mapprojection
notebooks/aster_with_bundle_adjust_and_jitter_correction
notebooks/lunar_recon_orbiter
notebooks/mars_mgs_orbital_camera
notebooks/mars_mro_ctx
notebooks/mars_mro_hirise
```
