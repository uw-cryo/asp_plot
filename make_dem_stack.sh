#! /bin/bash
#Generate per-pixel stats for stack of input DEMs
#Useful for evaluating relative precision for multi-view stack of pairwise DEMs after bundle_adjust

out_prefix=dem_composite

gdal_opt='-co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER'

dem_list=$(ls */*DEM.tif)
dem_fn_list=${out_prefix}_DEM_fn_list.txt
ls */*DEM.tif > $dem_fn_list

#Ortho potentially has different GSD than DEM
if [ ! -f ${out_prefix}_ortho-tile-0.tif ] ; then
    ortho_fn_list=${out_prefix}_ortho_fn_list.txt
    ls */*ortho.tif > $ortho_fn_list 
    dem_mosaic --tap -l $ortho_fn_list -o ${out_prefix}_ortho &
fi

#Assume tr and t_srs are uniform, use metadata from first DEM
#Start the weighted average mosaic
dem_mosaic --tap -l $dem_fn_list -o $out_prefix &
#Generate additional stats in parallel
stat_list="--median --count --stddev --nmad"
#for stat in $stat_list ; do dem_mosaic $stat --tap -l ${out_prefix}_fn_list.txt -o $out_prefix ; done
parallel --delay 2 "dem_mosaic {} --tap -l $dem_fn_list -o $out_prefix " ::: $stat_list

#Select "final" DEM, either weighted mean or median
dem_mos=${out_prefix}-tile-0.tif
#dem_mos=$out_prefix-tile-0-median.tif

#Generate hillshade
gdaldem hillshade $gdal_opt -compute_edges $dem_mos ${dem_mos%.*}_hs.tif

#Write anomaly maps
#parallel "compute_diff.py {} $dem_mos" ::: $dem_list

#Compute difference from reference
refdem='../COP30_lzw-adj_proj.tif'
compute_diff.py $dem_mos $refdem
