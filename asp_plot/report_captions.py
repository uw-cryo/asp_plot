"""Caption and description text for the ``asp_plot`` report sections.

These strings were previously inlined into the ~880-line ``cli/asp_plot.py``
``main()`` control flow. Pulling them into a data module keeps the report
pipeline (see :mod:`asp_plot.report_pipeline`) focused on orchestration and
makes the prose reviewable in one place. The body-dependent alignment
descriptions and the planetary captions are exposed as small template
functions; everything else is a module-level constant.
"""

# ---- Static section captions ----

# These were single string literals in the original ``main()``; kept as
# single-line literals here so the rendered captions stay byte-identical
# (flake8 E501 is ignored repo-wide, so long lines are fine).
INPUT_SCENES = "Left and right input scenes used for stereo processing. Non-mapprojected scenes are shown after ASP's alignment step (e.g., affineepipolar), which rotates images to create horizontal epipolar lines for correlation. Mapprojected scenes have been orthorectified with RPCs against a reference DEM to roughly align the two images prior to correlation, which reduces the disparity search range; they are displayed here in their map-projected orientation."

STEREO_GEOMETRY = "Stereo acquisition geometry skyplot and map view showing satellite viewing angles and scene footprints."

MATCH_POINTS = "Interest point matches between left and right images. These are produced by stereo_corr during its initial interest point matching step, which is used to set the search windows for subsequent dense correlation (not the dense correlation matches themselves)."

BUNDLE_RESIDUALS_LOG = (
    "Initial and final bundle adjustment residuals on a logarithmic scale."
)

BUNDLE_RESIDUALS_LINEAR = (
    "Initial and final bundle adjustment residuals on a linear scale."
)

MAP_PROJECTED_RESIDUALS = "Midpoint distance between final interest points projected onto the reference DEM used in processing."

GEODIFF = "Initial and final geodiff height differences compared to the reference DEM used in processing."

DISPARITY = "Horizontal and vertical disparity maps in pixels with quiver overlay."

DEM_RESULTS = "Output DEM with intersection error map and difference relative to the reference DEM used in processing."

DETAILED_HILLSHADE = "DEM hillshade. If the intersection error is available, zoomed subsets selected from low, medium, and high (left to right) uncertainty areas are displayed in the second row. If the mapprojected image is available, corresponding ortho image subsets are displayed in the bottom row."

# ---- ICESat-2 (Earth) captions ----

ICESAT2_MAP = "ICESat-2 ATL06-SR elevation differences vs. ASP DEM."

ICESAT2_HISTOGRAM = "Distribution of elevation differences between ICESat-2 ATL06-SR and ASP DEM with per-landcover statistics."

ICESAT2_PROFILE = (
    "Elevation profile along the ICESat-2 track with the "
    "most valid points, comparing ATL06-SR and DEM heights "
    "(top) and height differences (bottom), with a context "
    "hillshade map on the right. Blue/red highlights mark "
    "the 1 km segments with better and worse agreement "
    "between ICESat-2 and the DEM."
)

ICESAT2_SEGMENTS = (
    "1 km segments along the ICESat-2 track with better "
    "(left) and worse (right) agreement with the DEM. "
    "Segments are scored by 3·|median(dh)| + NMAD(dh), "
    "which weights the median bias three times more than "
    "the dispersion so that a segment with a large bias "
    "is not selected as 'better agreement' just because "
    "its NMAD is small."
)

ICESAT2_HISTOGRAM_ALIGNED = (
    "Pre- (steelblue) and post-alignment (orange) "
    "distributions of ICESat-2 minus DEM height "
    "differences, with per-landcover statistics in "
    "the two stacked text boxes. Box outline color "
    "matches the bar color."
)

ICESAT2_PROFILE_ALIGNED = (
    "Elevation profile along the ICESat-2 track after "
    "pc_align. The aligned DEM is overlaid on the "
    "profile and used to recompute the height "
    "differences shown in the lower panel."
)

ICESAT2_SEGMENTS_ALIGNED = (
    "The same better- and worse-agreement segments "
    "as above, now with the aligned DEM overlaid. "
    "Segment selection is held fixed so Median/NMAD "
    "can be compared directly."
)

EARTH_ALIGNMENT_DESCRIPTION = (
    "ASP's pc_align estimates a rigid 3D translation that "
    "minimizes the height residuals between the ASP DEM and "
    "ICESat-2 ATL06-SR ground-track points used as the "
    "reference point cloud. The translation is applied to "
    "the DEM directly (geotransform + pixel-value shift, no "
    "resampling) to produce the aligned DEM.\n\n"
    "Alignment Parameters (above):\n"
    "  - processing_level: ATL06-SR filter key used as the "
    "reference; 'all' uses every filtered point.\n"
    "  - minimum_points: minimum ATL06-SR point count "
    "required; fewer points skips the alignment.\n"
    "  - agreement_threshold: maximum relative disagreement "
    "across temporal sub-filters before the aligned DEM is "
    "flagged as inconsistent.\n"
    "  - min_translation_threshold: minimum translation "
    "magnitude (as a fraction of the DEM GSD) required to "
    "write out an aligned DEM.\n"
    "  - improvement_threshold_pct: minimum percentage "
    "reduction in p50 required to keep the aligned DEM on "
    "disk; below this, the aligned DEM is removed.\n\n"
    "Alignment Statistics (above, in meters):\n"
    "  - p16_beg / p50_beg / p84_beg: 16th / 50th / 84th "
    "percentile of the DEM-vs-ICESat absolute height "
    "residuals before alignment.\n"
    "  - p16_end / p50_end / p84_end: same percentiles "
    "after alignment.\n"
    "  - N_shift / E_shift / D_shift: north / east / down "
    "components of the applied translation vector.\n"
    "  - |T|: magnitude of the translation vector."
)


# ---- Planetary (Moon/Mars) caption templates ----


def planetary_altimetry_map(instrument):
    return f"{instrument} elevation differences vs. ASP DEM."


def planetary_altimetry_histogram(instrument):
    return f"Distribution of elevation differences between {instrument} and ASP DEM."


def planetary_altimetry_map_aligned(instrument):
    return (
        f"Pre- (left) and post-alignment (right) "
        f"map views of {instrument} elevation "
        f"differences. The aligned-DEM hillshade "
        f"is used as the backdrop for both panels."
    )


def planetary_altimetry_histogram_aligned(instrument):
    return (
        f"Pre- (steelblue) and post-alignment "
        f"(orange) distributions of {instrument} "
        f"minus DEM height differences with shared "
        f"bin edges."
    )


def planetary_alignment_description(instrument):
    return (
        f"ASP's pc_align estimates a rigid 3D translation that "
        f"minimizes the height residuals between the ASP DEM "
        f"and the {instrument} planetary radii. The CSV is "
        f"passed as the reference cloud with --csv-format "
        f"'1:lon 2:lat 3:radius_m', and --datum is set to "
        f"D_MARS or D_MOON to match the ASP DEM. The "
        f"resulting translation is applied to the DEM "
        f"directly (geotransform + pixel-value shift, no "
        f"resampling) to produce the aligned DEM.\n\n"
        f"Alignment Parameters (above):\n"
        f"  - max_displacement: pc_align upper bound on the "
        f"translation magnitude (m).\n"
        f"  - minimum_points: minimum {instrument} points "
        f"that overlap the DEM; below this the alignment is "
        f"skipped.\n"
        f"  - min_translation_threshold: minimum translation "
        f"magnitude (as a fraction of the DEM GSD) required "
        f"to write out an aligned DEM.\n"
        f"  - improvement_threshold_pct: minimum percentage "
        f"reduction in p50 required to keep the aligned DEM "
        f"on disk; below this, the aligned DEM is removed.\n\n"
        f"Alignment Statistics (above, in meters):\n"
        f"  - p16_beg / p50_beg / p84_beg: 16th / 50th / 84th "
        f"percentile of the DEM-vs-{instrument} absolute "
        f"height residuals before alignment.\n"
        f"  - p16_end / p50_end / p84_end: same percentiles "
        f"after alignment.\n"
        f"  - N_shift / E_shift / D_shift: north / east / down "
        f"components of the applied translation vector.\n"
        f"  - |T|: magnitude of the translation vector."
    )
