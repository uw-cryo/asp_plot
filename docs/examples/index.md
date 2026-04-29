# Example Notebooks

Examples of modular usage of `asp_plot`, organized by sensor type. Each notebook demonstrates the plotting classes and functions available for different satellite instruments.

## Earth-based

::::{grid} 1
:gutter: 3

:::{grid-item-card} WorldView — SpaceNet Atlanta
:link: notebooks/worldview_spacenet_atlanta_stereo
:link-type: doc

Stereo processing of publicly available SpaceNet Atlanta WorldView-2 data.
:::

:::{grid-item-card} WorldView — SpaceNet Atlanta (No Mapprojection)
:link: notebooks/worldview_spacenet_atlanta_stereo_without_mapprojection
:link-type: doc

Same Atlanta stereopair as the main notebook but processed without the mapprojection step (`--alignment-method affineepipolar`), with a side-by-side ICESat-2 histogram comparison.
:::

:::{grid-item-card} WorldView — SpaceNet Atlanta (Scene Selection)
:link: notebooks/worldview_spacenet_atlanta_stereo_scene_selection
:link-type: doc

Pair-ranking and DEM-vs-ICESat-2 comparison used to choose the Atlanta stereo pair processed in the companion notebook.
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

notebooks/worldview_spacenet_atlanta_stereo
notebooks/worldview_spacenet_atlanta_stereo_without_mapprojection
notebooks/worldview_spacenet_atlanta_stereo_scene_selection
notebooks/worldview_spacenet_ucsd_stereo
notebooks/worldview_spacenet_ucsd_stereo_scene_selection
notebooks/worldview_uyuni_jitter_plots
notebooks/aster_with_mapprojection
notebooks/aster_with_bundle_adjust_and_jitter_correction
notebooks/lunar_recon_orbiter
notebooks/mars_mgs_orbital_camera
notebooks/mars_mro_ctx
notebooks/mars_mro_hirise
```
