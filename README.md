# asp_plot
Scripts and notebooks to visualize output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

## Motivation
Our objective is to release a set of standalone Python scripts that can be run automatically on an ASP output directory to prepare a set of standard diagnostic plots, publication-quality output figures, and a pdf report with relevant information, similar to the reports prepared by many commercial SfM software packages (e.g., Agisoft Metashape, Pix4DMapper).

## Status
This is a work in progress, with initial notebooks compiled from recent projects using sample stereo images from the Maxar WorldView, Planet SkySat-C and BlackSky Global constellations. We plan to refine these tools in the coming year, and we welcome community contributions and input. 

## Ben Temporary Notes:

I'm envisioning a command line tool that outputs a report with:
Overview plots:

  Satellite geometry plot (modified or incorporated dg_geom_plot.py script)
  
  Images used (just stereo / two pair for now, but could extend to multi-view)

Bundle Adjust plots:

  initial and final residuals plotted on some background imagery
  
  optionally if mapproj-dem was used then plots of final residuals in metric units
  
  optionally if there was a pc_align step then a plot of before / after residuals
  
  optionally if geodiff tool was run, then plot of initial and final vertical difference (initial/final_residuals_pointmap-diff.csv)

Stereo plots
  
  vwip interest points on top of aligned (i.e. map-projected) L and R images (probably just use L_sub / R_sub)
  
  L__R.match points on top of L_sub.tif image (don't see the need to plot full size L.tif)
  
  D_sub.tif? Or RD.tif? Or F.tif? Benefit of D_sub is small size and not a virtual file, so not dependent on the thousands of sub-folders
  
    Show three disparity plots for horizontal, vertical, and magnitude?
  
  DEM (overlain on HS) + IntErr; if DEMs produced at multiple postings then a separate plot for each
  
    Potentially an inset zoom image(s) to get a closer look at the hillshade

  Difference map of DEM with command-line passed reference DEMs (could be COP30 always, but option to pass additional local DEMs, like 3DEP)

  The figures are written into a markdown document and that markdown is converted to PDF at the end (maybe keeping the markdown around for manual editing, or just clean this up as well)

More things like metrics and descriptions of plots and exact commands that were run to output each step can be inserted into the markdown document as well. There might also be additional control flow and options for plots specific to pinhole cameras. But the framework of the command line tool would hopefully be easily re-usable and extendable.
