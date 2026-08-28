#!/usr/bin/env bash
# Atlanta same-pass WorldView-2 scene-combination benchmark runs (uw-cryo/asp_plot#169).
#
#   usage: run_benchmark.sh /path/to/atlanta_mvs
#
# Extends the runs behind worldview_spacenet_atlanta_mvs.ipynb (5-scene
# bundle_adjust, 3- and 5-scene MVS, pairs 13-10 / 13-16 / 10-16 and their
# dem_mosaic) to the full matrix scored by worldview_spacenet_atlanta_benchmark.ipynb:
#
#   1. the seven remaining pairs           -> all ten pairs, convergence 5-32 deg
#   2. dem_mosaic of all ten pairs, and of the six pairs with convergence > 15 deg
#   3. a nested quad {13,10,16,21}          -> the 2->3->4->5 scene curve
#      (pair 13-10 < MVS3 {13,10,16} < quad < MVS5) and a wide-spread triple {13,8,21}
#
# Every step is skipped when its product already exists, so the script can be
# re-run after an interruption. Settings are identical to run_mvs.sh /
# run_pairwise.sh (asp_mgm, subpixel-mode 9, affineepipolar, the shared 5-scene
# ba/run prefix, point2dem 1.9 m EPSG:32616 with --errorimage).
#
# Cost on an 8-thread laptop: pair ~1 h, triple ~2 h, quad ~2.5 h -> about 11-12 h,
# ~30 GB. wv_correct is ~40 s per scene.
#
# Scenes (2009-12-22, SpaceNet AOI_6_Atlanta; along-track order 10, 8, 13, 16, 21):
#   nadir8   10300100023BC100_P001
#   nadir10  1030010003CAF100_P002
#   nadir13  1030010002B7D800_P002   (MVS reference; the known crop window)
#   nadir16  1030010002649200_P001
#   nadir21  1030010003127500_P001
#
# Left-image crop windows: nadir13's is the reference crop used by every existing
# run; the others were derived from the bundle_adjust clean match points that fall
# inside that crop, padded 100 px (the benchmark notebook reproduces the derivation;
# nadir10's reproduces the window run_pairwise.sh used).
set -euo pipefail

DATA=${1:?usage: run_benchmark.sh /path/to/atlanta_mvs}
cd "$DATA"
# Append, never prepend: the ASP release bundles its own python.
export PATH="$PATH:${ASP_BIN:-$HOME/asp/dev/bin}"
THREADS=${THREADS:-8}
# The raw nadir13 / nadir10 L1B tiles live with the earlier two-scene example.
RAW_FALLBACK_DIR=${RAW_FALLBACK_DIR:-../atlanta_stereo_22deg_0d}

N8=10300100023BC100_P001
N10=1030010003CAF100_P002
N13=1030010002B7D800_P002
N16=1030010002649200_P001
N21=1030010003127500_P001

crop_win () {
    case "$1" in
        "$N13") echo "5879 13107 12981 11894" ;;
        "$N10") echo "5540 13223 13206 12931" ;;
        "$N8")  echo "5461 9789 13354 12980" ;;
        "$N16") echo "6083 8756 12879 11461" ;;
        "$N21") echo "6467 8177 12495 10687" ;;
        *) echo "unknown scene $1" >&2; exit 1 ;;
    esac
}

log () { echo "=== [$(date)] $*"; }

# 0. CCD correction. The *_corr.tif intermediates were purged after the earlier
#    runs (two of them were symlinks into the two-scene example, now dangling).
for cid in $N13 $N10 $N8 $N16 $N21; do
    if [ ! -s "${cid}_corr.tif" ]; then
        raw="${cid}.tif"
        [ -f "$raw" ] || raw="${RAW_FALLBACK_DIR}/${cid}.tif"
        log "wv_correct $cid (from $raw)"
        rm -f "${cid}_corr.tif"
        wv_correct --threads "$THREADS" "$raw" "${cid}.xml" "${cid}_corr.tif"
    fi
done

# run_stereo <outdir> <left> [<right> ...]   -- first scene is the reference
run_stereo () {
    local out=$1; shift
    local left=$1
    local imgs="" cams=""
    for cid in "$@"; do
        imgs="$imgs ${cid}_corr.tif"
        cams="$cams ${cid}.xml"
    done
    log "stereo $out ($*)"
    if [ ! -f "$out/run-PC.tif" ]; then
        # shellcheck disable=SC2086
        parallel_stereo --stereo-algorithm asp_mgm --subpixel-mode 9 \
            --alignment-method affineepipolar \
            --left-image-crop-win $(crop_win "$left") \
            --bundle-adjust-prefix ba/run \
            $imgs $cams "$out/run"
    fi
    if [ ! -f "$out/run-DEM.tif" ]; then
        point2dem --tr 1.9 --t_srs EPSG:32616 --errorimage "$out/run-PC.tif"
    fi
}

# mosaic <prefix> <dem> [<dem> ...]
mosaic () {
    local prefix=$1; shift
    if [ ! -f "${prefix}-DEM.tif" ]; then
        log "dem_mosaic $prefix"
        dem_mosaic "$@" -o "$prefix"
        mv "${prefix}-tile-0.tif" "${prefix}-DEM.tif"
    fi
}

# 1. The seven remaining pairs (13-10, 13-16 and 10-16 exist from run_pairwise.sh).
#    Left image = the reference-side scene so the known crops are reused where possible.
run_stereo stereo_pair_13_8  "$N13" "$N8"
run_stereo stereo_pair_13_21 "$N13" "$N21"
run_stereo stereo_pair_10_8  "$N10" "$N8"
run_stereo stereo_pair_10_21 "$N10" "$N21"
run_stereo stereo_pair_8_16  "$N8"  "$N16"
run_stereo stereo_pair_8_21  "$N8"  "$N21"
run_stereo stereo_pair_16_21 "$N16" "$N21"

# 2. Pairwise + mosaic at five scenes: every pair, and only the well-converged ones
#    (> 15 deg: 13-8 16.3, 8-16 21.5, 13-10 21.8, 8-21 26.8, 10-16 26.9, 10-21 32.3).
ALL10="stereo_pair_13_10 stereo_pair_13_16 stereo_pair_10_16 stereo_pair_13_8 stereo_pair_13_21 \
       stereo_pair_10_8 stereo_pair_10_21 stereo_pair_8_16 stereo_pair_8_21 stereo_pair_16_21"
WIDE6="stereo_pair_13_8 stereo_pair_8_16 stereo_pair_13_10 stereo_pair_8_21 stereo_pair_10_16 stereo_pair_10_21"
# shellcheck disable=SC2046
mosaic pairwise10_mosaic    $(for d in $ALL10; do echo "$d/run-DEM.tif"; done)
# shellcheck disable=SC2046
mosaic pairwise_wide6_mosaic $(for d in $WIDE6; do echo "$d/run-DEM.tif"; done)

# 3. Scene count: the nested quad, and a triple that spans the pass (16.3 / 10.5 / 26.8 deg).
run_stereo stereo_mvs4      "$N13" "$N10" "$N16" "$N21"
run_stereo stereo_mvs3_wide "$N13" "$N8"  "$N21"

log "ALL DONE"
