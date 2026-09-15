#!/usr/bin/env bash
# UCSD (CORE3D WorldView-3) second site of the scene-combination benchmark
# (uw-cryo/asp_plot#169): five multi-date winter scenes over a 3 x 3 km crop of
# Mount Soledad, processed as one multi-view run per reference scene and as all
# ten pairs plus dem_mosaic blends, then scored in
# worldview_spacenet_benchmark.ipynb next to the Atlanta results.
#
#   bash notebooks/WorldView/run_ucsd_benchmark.sh ~/Desktop/asp-plot-examples/ucsd_mvs
#
# Every step is skipped when its product exists. Needs ASP on PATH (appended,
# never prepended: the release bundles its own python) and `aws` for the
# public S3 downloads. Measured on an 8-thread laptop (2026-09-13): 50 min for
# the joint bundle_adjust, 2 h 37 min for the first five-scene run, 40-45 min
# per pair, 2 h 50 min and 3 h 55 min for the two other five-scene runs,
# 16 h 20 min end to end; 77 GB on disk before the F/RD/R intermediates are
# removed.
#
# The scenes are JPEG2000-compressed NITF. ASP 3.8.0-alpha ships GDAL's
# JP2OpenJPEG driver as a plugin it does not load by itself, so GDAL_DRIVER_PATH
# points at the ASP lib directory for the one conversion to GeoTIFF; nothing
# downstream touches the NITF again. One converted scene inherited a lat/lon
# geotransform from the NITF corner coordinates, which asp_plot's RPC reader
# takes as "already map-projected" and refuses, hence the gdal_edit step.
set -euo pipefail
cd "${1:?usage: run_ucsd_benchmark.sh <work dir>}"
ASP_ROOT=$(dirname "$(dirname "$(command -v parallel_stereo)")")
export PROJ_DATA="$ASP_ROOT/share/proj"
T=8
log () { echo; echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 1. Scenes: five winter 2014-15 collects, sun elevation 30-40 deg, chosen in the
#    notebook from the 35 CORE3D WV3 scenes for a spread of off-nadir angle and
#    azimuth. Reference-star convergence (see the notebook): nadir 9/15/17/26,
#    n24 17/25/26/34, n08 9/17/19/34.
S3=s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/UCSD/WV3/PAN/
SCENES="09NOV14WV031300014NOV09182948-P1BS-500647758070_01_P001_________AAE_0AAAAABPABT0
28NOV14WV031200014NOV28182901-P1BS-500647760090_01_P001_________AAE_0AAAAABPABQ0
23DEC14WV031300014DEC23182236-P1BS-500647759010_01_P001_________AAE_0AAAAABPABP0
24JAN15WV031300015JAN24183547-P1BS-500647760080_01_P001_________AAE_0AAAAABPABR0
12FEB15WV031300015FEB12183926-P1BS-500647760030_01_P001_________AAE_0AAAAABPABS0"
mkdir -p images
log "download + convert"
for b in $SCENES; do
    for e in NTF tar; do [ -s "images/$b.$e" ] || aws s3 --no-sign-request cp "$S3$b.$e" "images/$b.$e" --only-show-errors; done
    mkdir -p "images/meta/$b"; tar xf "images/$b.tar" -C "images/meta/$b"
    xml=$(find "images/meta/$b" -iname "*P1BS*P001.XML" | head -1)
    catid=$(grep -o "<CATID>[^<]*" "$xml" | sed 's/<CATID>//')
    [ -s "${catid}_P001.xml" ] || cp "$xml" "${catid}_P001.xml"
    if [ ! -s "${catid}_P001.tif" ]; then
        GDAL_DRIVER_PATH="$ASP_ROOT/lib" gdal_translate -q -co TILED=YES -co COMPRESS=DEFLATE \
            -co BIGTIFF=YES -co NUM_THREADS=ALL_CPUS "images/$b.NTF" "${catid}_P001.tmp.tif"
        gdal_edit.py -unsetgt -a_srs "" "${catid}_P001.tmp.tif"
        mv "${catid}_P001.tmp.tif" "${catid}_P001.tif"
    fi
done
NADIR=104001000496A100_P001; N15=10400100047BBB00_P001; N24=10400100057DD500_P001
N16=10400100071D8800_P001; N08=1040010007A93700_P001

# 2. Crop A (Mount Soledad), UTM 11N 476000-479000 E / 3632000-3635000 N,
#    projected through each scene's RPC at -40 and 220 m and padded 100 px.
#    (asp_plot's rasterio RPCTransformer does this; the windows are recorded
#    here so the run does not depend on the helper.)
win () { case "$1" in
    $NADIR) echo "10437 30472 9969 9968" ;;  $N15) echo "10775 28462 9644 9559" ;;
    $N24) echo "11951 29190 8733 9754" ;;    $N16) echo "10953 28317 9568 9572" ;;
    $N08) echo "10767 30242 9845 9811" ;;    esac; }
mkdir -p cropA
ALL="$NADIR $N15 $N24 $N16 $N08"

# 3. One bundle adjustment of the full scenes, shared by every run below.
log "bundle_adjust"
[ -f cropA/ba/run-${N08}.adjust ] || bundle_adjust --threads $T --ip-per-image 10000 \
    --tri-weight 0.1 --tri-robust-threshold 0.1 --camera-weight 0 \
    $(for c in $ALL; do echo "$c.tif"; done) $(for c in $ALL; do echo "$c.xml"; done) -o cropA/ba/run

# stereo <outdir> <left> [<right> ...] -- the left scene is the reference and
# gives the crop window; same settings as the Atlanta runs at 1.2 m (~4x GSD).
stereo () {
    local out=$1; shift; local left=$1
    local imgs="" cams=""
    for c in "$@"; do imgs="$imgs $c.tif"; cams="$cams $c.xml"; done
    log "$out ($*)"
    # shellcheck disable=SC2086
    [ -f "$out/run-PC.tif" ] || parallel_stereo --threads-singleprocess $T \
        --stereo-algorithm asp_mgm --subpixel-mode 9 --alignment-method affineepipolar \
        --left-image-crop-win $(win "$left") --bundle-adjust-prefix cropA/ba/run \
        $imgs $cams "$out/run"
    [ -f "$out/run-DEM.tif" ] || point2dem --threads $T --tr 1.2 --t_srs EPSG:32611 --errorimage "$out/run-PC.tif"
}

# 4. Five scenes, three references: a multi-view run is a star of
#    reference-to-scene pairs, so the reference is the experiment.
stereo cropA/stereo_mvs       $NADIR $N15 $N24 $N16 $N08
stereo cropA/stereo_mvs_ref24 $N24 $NADIR $N15 $N16 $N08
stereo cropA/stereo_mvs_ref08 $N08 $NADIR $N15 $N24 $N16

# 5. All ten pairs (scene index order 01..05 = nadir, n15, n24, n16, n08).
stereo cropA/stereo_pair_01_02 $NADIR $N15;  stereo cropA/stereo_pair_01_03 $NADIR $N24
stereo cropA/stereo_pair_01_04 $NADIR $N16;  stereo cropA/stereo_pair_01_05 $NADIR $N08
stereo cropA/stereo_pair_02_03 $N15 $N24;    stereo cropA/stereo_pair_02_04 $N15 $N16
stereo cropA/stereo_pair_02_05 $N15 $N08;    stereo cropA/stereo_pair_03_04 $N24 $N16
stereo cropA/stereo_pair_03_05 $N24 $N08;    stereo cropA/stereo_pair_04_05 $N16 $N08

# 6. Blends: every pair (default average, median, and the count/stddev/nmad
#    agreement layers) and the eight pairs above 15 deg (drops 01_05 at 9.2 and
#    02_04 at 8.0 deg).
mosaic () {  # mosaic <prefix> <extra dem_mosaic flag or ""> <dem> ...
    local prefix=$1 flag=$2; shift 2
    local tag=${flag:---DEM}; tag=${tag#--}
    [ -f "$prefix-tile-0-$tag.tif" ] || [ -f "$prefix-DEM.tif" -a -z "$flag" ] || \
        dem_mosaic --threads $T --tap $flag "$@" -o "$prefix"
    [ -z "$flag" ] && [ -f "$prefix-tile-0.tif" ] && mv "$prefix-tile-0.tif" "$prefix-DEM.tif"
    return 0
}
ALL10=$(for p in 01_02 01_03 01_04 01_05 02_03 02_04 02_05 03_04 03_05 04_05; do echo "cropA/stereo_pair_$p/run-DEM.tif"; done)
WIDE8=$(for p in 01_02 01_03 01_04 02_03 02_05 03_04 03_05 04_05; do echo "cropA/stereo_pair_$p/run-DEM.tif"; done)
log "dem_mosaic"
for flag in "" --count --median --stddev --nmad; do mosaic cropA/pairwise_mosaic "$flag" $ALL10; done
for flag in "" --median; do mosaic cropA/pairwise_wide8_mosaic "$flag" $WIDE8; done
log "DONE"
