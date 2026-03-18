import sys

import click

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
      3. Download the .zip file from the link in the email
      4. Unzip and pass the *_topo_csv.csv to asp_plot via --altimetry_csv

    \b
    Example:
      request_planetary_altimetry --dem stereo/output-DEM.tif --email user@example.com
      # ... wait for email, download zip ...
      asp_plot --directory ./ --altimetry_csv /path/to/*_topo_csv.csv
    """
    body = detect_planetary_body(dem)

    if body == "earth":
        click.echo(
            "\nThis DEM is an Earth DEM. ICESat-2 altimetry is fetched "
            "automatically by asp_plot — no separate request needed.\n"
            "Run: asp_plot --directory <dir> --plot_altimetry True\n"
        )
        sys.exit(0)

    bounds = get_planetary_bounds(dem, body=body)
    click.echo(f"\nDetected body: {body}")
    click.echo(
        f"DEM bounds (0-360 lon): "
        f"lon [{bounds['westernlon']:.4f}, {bounds['easternlon']:.4f}], "
        f"lat [{bounds['minlat']:.4f}, {bounds['maxlat']:.4f}]"
    )

    if body == "moon":
        _submit_lola(bounds, email, channels)
    elif body == "mars":
        _submit_mola(bounds, email)


def _submit_lola(bounds, email, channels):
    """Submit an async LOLA RDR query."""
    from asp_plot.altimetry import Altimetry

    click.echo(f"\nSubmitting LOLA query (channels={channels}) ...")
    try:
        job_id = Altimetry._gds_query_async(
            "lolardr",
            bounds,
            "u",
            email=email,
            channel=channels,
        )
        click.echo("\nLOLA query submitted successfully!")
        click.echo(f"  Job ID: {job_id}")
        click.echo(f"  Email:  {email}")
        click.echo(
            "\nYou will receive an email when the data is ready. "
            "Download and unzip the result, then pass the *_topo_csv.csv to asp_plot:\n"
            "  asp_plot --directory <dir> --altimetry_csv <*_topo_csv.csv>\n"
        )
    except Exception as e:
        click.echo(f"\nError submitting LOLA query: {e}", err=True)
        sys.exit(1)


def _submit_mola(bounds, email):
    """Submit an async MOLA PEDR query."""
    from asp_plot.altimetry import Altimetry

    click.echo("\nSubmitting MOLA query ...")
    try:
        job_id = Altimetry._gds_query_async(
            "molapedr",
            bounds,
            "v",
            email=email,
        )
        click.echo("\nMOLA query submitted successfully!")
        click.echo(f"  Job ID: {job_id}")
        click.echo(f"  Email:  {email}")
        click.echo(
            "\nYou will receive an email when the data is ready. "
            "Download and unzip the result, then pass the *_topo_csv.csv to asp_plot:\n"
            "  asp_plot --directory <dir> --altimetry_csv <*_topo_csv.csv>\n"
        )
    except Exception as e:
        click.echo(f"\nError submitting MOLA query: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
