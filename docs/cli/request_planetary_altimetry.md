# request_planetary_altimetry

Submit a LOLA (Moon) or MOLA (Mars) altimetry data request for a planetary DEM. This tool auto-detects the planetary body from the DEM's CRS and submits an asynchronous query to the [ODE Granular Data System (GDS)](https://oderest.rsl.wustl.edu/) REST API.

## Workflow

Planetary altimetry data requires an asynchronous request/download workflow because the ODE GDS API processes queries in a queue:

1. **Submit request** — run `request_planetary_altimetry` with your DEM and email
2. **Wait for email** — you will receive a download link (may take minutes to hours)
3. **Download and unzip** — extract the `*_topo_csv.csv` file from the zip
4. **Generate report** — pass the CSV to `asp_plot` via `--altimetry_csv`

## Basic usage

```bash
request_planetary_altimetry \
  --dem stereo/output-DEM.tif \
  --email user@example.com
```

Then, once the data is downloaded and unzipped:

```bash
asp_plot --directory ./ \
         --stereo_directory stereo \
         --altimetry_csv /path/to/MolaPEDR_*_topo_csv.csv \
         --add_basemap False \
         --plot_geometry False
```

## LOLA channels

For Moon DEMs, the `--channels` option controls which LOLA detector channels are included. LOLA has 5 detectors that fire simultaneously in a cross pattern. Including fewer channels produces smaller, faster queries:

```bash
# Channel 1 only (default, fastest)
request_planetary_altimetry --dem run/run-DEM.tif --email user@example.com --channels tffff

# All 5 channels (more points, slower query)
request_planetary_altimetry --dem run/run-DEM.tif --email user@example.com --channels ttttt
```

## Request metadata

The tool saves an `altimetry_request_info.yml` file alongside the DEM containing the job ID, query parameters, bounding box, and submission timestamp.

## Full options

```
Usage: request_planetary_altimetry [OPTIONS]

  Submit a LOLA or MOLA altimetry data request for a planetary DEM.

  Auto-detects the planetary body from the DEM's CRS and submits an
  asynchronous query to the ODE Granular Data System (GDS) REST API.

  Workflow:
    1. Run this command with your DEM and email
    2. Wait for the email notification (may take minutes to hours)
    3. Download and unzip the result
    4. Pass the *_topo_csv.csv to asp_plot via --altimetry_csv

  Example:
    request_planetary_altimetry --dem stereo/output-DEM.tif --email user@example.com
    # ... wait for email, download and unzip ...
    asp_plot --directory ./ --altimetry_csv /path/to/*_topo_csv.csv

Options:
  --dem PATH       Path to the ASP DEM file. The planetary body (Moon/Mars) is
                   auto-detected from the CRS.  [required]
  --email TEXT     Email address for notification when the query finishes. You
                   will receive a download link.  [required]
  --channels TEXT  LOLA detector channels to include (Moon only). 5
                   characters, t=include/f=exclude. Default: tffff (channel 1
                   only, for faster queries). Use ttttt for all 5 channels.
  --help           Show this message and exit.
```
