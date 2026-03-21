import os
import sys
from datetime import datetime, timezone

import click
import yaml

from asp_plot.utils import detect_planetary_body, get_planetary_bounds


@click.command()
@click.option(
    "--dem",
    required=True,
    type=click.Path(exists=True),
    help="Path to the ASP DEM file. The planetary body (Moon/Mars) is auto-detected from the CRS.",
)
@click.option(
    "--email",
    required=True,
    help="Email address for notification when the query finishes. You will receive a download link.",
)
@click.option(
    "--channels",
    default="tffff",
    help="LOLA detector channels to include (Moon only). 5 characters, t=include/f=exclude. Default: tffff (channel 1 only, for faster queries). Use ttttt for all 5 channels.",
)
def main(dem, email, channels):
    """Submit a LOLA or MOLA altimetry data request for a planetary DEM.

    Auto-detects the planetary body from the DEM's CRS and submits an
    asynchronous query to the ODE Granular Data System (GDS) REST API.

    \b
    Workflow:
      1. Run this command with your DEM and email
      2. Wait for the email notification (may take minutes to hours)
      3. Download and unzip the result
      4. Pass the *_topo_csv.csv to asp_plot via --altimetry_csv

    \b
    Example:
      request_planetary_altimetry --dem stereo/output-DEM.tif --email user@example.com
      # ... wait for email, download and unzip ...
      asp_plot --directory ./ --altimetry_csv /path/to/*_topo_csv.csv
    """
    from asp_plot.altimetry import GDS_BASE_URL, gds_query_async

    body = detect_planetary_body(dem)

    if body == "earth":
        click.echo(
            "\nThis DEM is an Earth DEM. ICESat-2 altimetry is fetched "
            "automatically by asp_plot — no separate request needed.\n"
            "Run: asp_plot --directory <dir> --plot_altimetry True\n"
        )
        sys.exit(0)

    bounds = get_planetary_bounds(dem, body=body)
    dem_abs = os.path.abspath(dem)

    click.echo(f"\nDetected body: {body}")
    click.echo(
        f"DEM bounds (0-360 lon): "
        f"lon [{bounds['westernlon']:.4f}, {bounds['easternlon']:.4f}], "
        f"lat [{bounds['minlat']:.4f}, {bounds['maxlat']:.4f}]"
    )

    # Build query parameters
    if body == "moon":
        query_type = "lolardr"
        results_code = "u"
        instrument = "LOLA"
        extra_params = {"channel": channels}
    else:
        query_type = "molapedr"
        results_code = "v"
        instrument = "MOLA"
        extra_params = {}

    click.echo(f"\nSubmitting {instrument} query ...")
    try:
        job_id = gds_query_async(
            query_type,
            bounds,
            results_code,
            email=email,
            **extra_params,
        )
    except Exception as e:
        click.echo(f"\nError submitting {instrument} query: {e}", err=True)
        sys.exit(1)

    # Save request metadata alongside the DEM
    info = {
        "job_id": job_id,
        "instrument": instrument,
        "body": body,
        "api_endpoint": GDS_BASE_URL,
        "query_type": query_type,
        "results_code": results_code,
        "email": email,
        "dem": dem_abs,
        "bounds": {
            "westernlon": round(float(bounds["westernlon"]), 6),
            "easternlon": round(float(bounds["easternlon"]), 6),
            "minlat": round(float(bounds["minlat"]), 6),
            "maxlat": round(float(bounds["maxlat"]), 6),
        },
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
    }
    if body == "moon":
        info["channels"] = channels

    info_path = os.path.join(os.path.dirname(dem_abs), "altimetry_request_info.yml")
    with open(info_path, "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)

    click.echo(f"\n{instrument} query submitted successfully!")
    click.echo(f"  Job ID:   {job_id}")
    click.echo(f"  Email:    {email}")
    click.echo(f"  Saved:    {info_path}")
    click.echo(
        "\nYou will receive an email when the data is ready. "
        "Download and unzip the result, then pass the *_topo_csv.csv to asp_plot:\n"
        "  asp_plot --directory <dir> --altimetry_csv <*_topo_csv.csv>\n"
    )


if __name__ == "__main__":
    main()
