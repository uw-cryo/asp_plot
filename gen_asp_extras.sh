 #Notes for script to run after parallel_stereo to generate sidecar files for plotting and diagnostics
 
 #Create difference map with refdem
 compute_diff.py *DEM.tif $refdem
 
 #Can run stereo_pprc again
 #--stddev-mask-thresh -1 --stddev-mask-kernel 7
 L_stddev_filter_output.tif
 
 #Generate point density and std, or use PDAL
 point2dem --filter count $point2dem_opt stereo_${ba_prefix}-PC.tif
 point2dem --filter stddev $point2dem_opt stereo_${ba_prefix}-PC.tif
 
 #Generate hillshade
 #Try combined and multidirectional as well
 #https://gdal.org/programs/gdaldem.html
 gdaldem hillshade $gdal_opt -compute_edges stereo_${ba_prefix}-DEM.tif stereo_${ba_prefix}-DEM_hs.tif
 
 #Generate correlation scores - needs to be consistent with the original correlation method
 corr_eval stereo_${ba_prefix}-L.tif stereo_${ba_prefix}-R.tif stereo_${ba_prefix}-F.tif stereo_${ba_prefix}