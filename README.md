# asp_plot
Scripts and notebooks to visualize output from the [NASA Ames Stereo Pipeline (ASP)](https://github.com/NeoGeographyToolkit/StereoPipeline).

## Motivation
Our objective is to release a set of standalone Python scripts that can be run automatically on an ASP output directory to prepare a set of standard diagnostic plots, publication-quality output figures, and a pdf report with relevant information, similar to the reports prepared by many commercial SfM software packages (e.g., Agisoft Metashape, Pix4DMapper).

## Status
This is a work in progress, with initial notebooks compiled from recent projects using sample stereo images from the Maxar WorldView, Planet SkySat-C and BlackSky Global constellations. We plan to refine these tools in the coming year, and we welcome community contributions and input. 

## Notes on tests

The ASP uses this tarball for running tests (568 MB size):

```
wget https://github.com/NeoGeographyToolkit/StereoPipelineTest/releases/download/0.0.1/StereoPipelineTest.tar
```

[As specified here, it contains all tests scripts, **data**, and expected results.](https://github.com/NeoGeographyToolkit/StereoPipeline/blob/df18be8d32435f7f9db829b9fc951f59bda86e55/.github/workflows/build_test.sh#L200-L206)

The `data/` directory in that tarball looks like:

```
8.9M - 13APR25_WV02_SEVZ_1030010021A8A100_10030010021A64500_DEM3-3m_10pct.tif
 55M - B17_016219_1978_XN_17N282W.8bit.cub
 37K - B17_016219_1978_XN_17N282W.8bit.json
 55M - B18_016575_1978_XN_17N282W.8bit.cub
 41K - B18_016575_1978_XN_17N282W.8bit.json
3.1M - Copernicus_DSM.tif
120M - M181058717LE.ce.cub
171K - M181058717LE.ce.json
120M - M181073012LE.ce.cub
176K - M181073012LE.ce.json
6.3M - Severnaya-Bedrock-UTM47-Ellipsoidal-Height.txt
5.5M - basic_panchromatic.tif
 20M - ref-mars-PC.tif
147K - ref-mars_yxz.csv
```

Can we use that data in the tests here as well? Or perhaps better to keep separate, some useful files maybe? (But probably not the `.cub` or `.json` files).
