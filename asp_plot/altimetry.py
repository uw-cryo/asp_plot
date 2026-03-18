import json
import logging
import os
from datetime import datetime, timedelta, timezone

import contextily as ctx
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from rasterio import plot as rioplot
from sliderule import sliderule as sliderule_api

from asp_plot.alignment import Alignment
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import ColorBar, Raster, glob_file, save_figure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

ICESAT2_MISSION_START = datetime(2018, 10, 14, tzinfo=timezone.utc)

WORLDCOVER_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse",
    70: "Snow/ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss/lichen",
}


def _nmad(a, c=1.4826):
    """Normalized Median Absolute Deviation."""
    return np.nanmedian(np.fabs(a - np.nanmedian(a))) * c


class Altimetry:
    """
    Process and analyze ICESat-2 ATL06-SR altimetry data with ASP DEMs.

    This class provides functionality to request, filter, and analyze
    ICESat-2 ATL06-SR altimetry data in conjunction with ASP-generated
    DEMs. It includes methods to request data from the SlideRule API,
    filter based on various criteria, align DEMs to altimetry data,
    and visualize the results.

    Attributes
    ----------
    directory : str
        Root directory for outputs and analysis
    dem_fn : str
        Path to the DEM file to analyze
    aligned_dem_fn : str or None
        Path to the aligned DEM file if available
    atl06sr_processing_levels : dict
        Dictionary of ATL06-SR data for different processing levels
    atl06sr_processing_levels_filtered : dict
        Dictionary of filtered ATL06-SR data for different processing levels
    alignment_report_df : pandas.DataFrame or None
        DataFrame containing alignment reports when available

    Examples
    --------
    >>> altimetry = Altimetry('/path/to/directory', '/path/to/dem.tif')
    >>> altimetry.request_atl06sr_multi_processing(save_to_parquet=True)
    >>> altimetry.filter_esa_worldcover(filter_out="water")
    >>> altimetry.alignment_report()
    >>> altimetry.mapview_plot_atl06sr_to_dem()
    """

    def __init__(
        self,
        directory,
        dem_fn,
        aligned_dem_fn=None,
        atl06sr_processing_levels={},
        atl06sr_processing_levels_filtered={},
        # atl03sr=None,
        **kwargs,
    ):
        """
        Initialize the Altimetry object.

        Parameters
        ----------
        directory : str
            Root directory for outputs and analysis
        dem_fn : str
            Path to the DEM file to analyze
        aligned_dem_fn : str, optional
            Path to an already aligned DEM file, default is None
        atl06sr_processing_levels : dict, optional
            Pre-loaded ATL06-SR data for different processing levels, default is empty dict
        atl06sr_processing_levels_filtered : dict, optional
            Pre-loaded filtered ATL06-SR data, default is empty dict
        **kwargs : dict, optional
            Additional keyword arguments for future extensions

        Raises
        ------
        ValueError
            If the DEM file or aligned DEM file (if provided) does not exist

        Notes
        -----
        Initializes a SlideRule session, which requires an active internet connection.
        This can be modified to work offline with pre-downloaded datasets.
        """
        self.directory = directory

        if not os.path.exists(dem_fn):
            raise ValueError(f"DEM file not found: {dem_fn}")
        self.dem_fn = dem_fn

        if aligned_dem_fn is not None and not os.path.exists(aligned_dem_fn):
            raise ValueError(f"Aligned DEM file not found: {aligned_dem_fn}")
        self.aligned_dem_fn = aligned_dem_fn

        self.atl06sr_processing_levels = atl06sr_processing_levels
        self.atl06sr_processing_levels_filtered = atl06sr_processing_levels_filtered

        # Lazy SlideRule initialization — only needed for ICESat-2 methods
        self._sliderule_initialized = False

        # Planetary altimetry data (LOLA/MOLA)
        self.planetary_points = None

        # TODO: Implement alongside request_atl03sr below
        # if atl03sr is not None and not isinstance(atl03sr, gpd.GeoDataFrame):
        #     raise ValueError("ATL03 must be a GeoDataFrame if provided.")
        # self.atl03sr = atl03sr

    # TODO: Implement ATL03 pull via x-series API: sliderule_api.run("atl03x", parms)
    # without the "fit" key to get photon-level data. Warning: this returns a very
    # large GeoDataFrame and should only be used for targeted profile visualizations.

    def _ensure_sliderule(self):
        """Initialize the SlideRule session on first use."""
        if not self._sliderule_initialized:
            sliderule_api.init("slideruleearth.io", verbose=True)
            self._sliderule_initialized = True

    # --- ODE GDS REST API client methods (for LOLA/MOLA) ---

    GDS_BASE_URL = "https://oderest.rsl.wustl.edu/livegds"

    @staticmethod
    def _gds_query_async(query_type, bounds, results_code, email=None, **extra_params):
        """Submit an async query to the ODE GDS REST API.

        Parameters
        ----------
        query_type : str
            GDS query type, e.g. ``"lolardr"`` or ``"molapedr"``.
        bounds : dict
            Dictionary with ``westernlon``, ``easternlon``, ``minlat``,
            ``maxlat`` keys.
        results_code : str
            GDS results format code (e.g. ``"u"`` for LOLA, ``"v"`` for MOLA).
        email : str or None, optional
            Email for notification when query finishes.
        **extra_params
            Additional GDS query parameters (e.g. ``channel="ttttt"``).

        Returns
        -------
        str
            Job ID for polling.
        """
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        params = {
            "query": query_type,
            "results": results_code,
            "westernlon": bounds["westernlon"],
            "easternlon": bounds["easternlon"],
            "minlat": bounds["minlat"],
            "maxlat": bounds["maxlat"],
            "async": "t",
        }
        if email:
            params["email"] = email
        params.update(extra_params)

        url = f"{Altimetry.GDS_BASE_URL}?{urllib.parse.urlencode(params)}"
        logger.info(f"GDS async query: {url}")
        print(f"Submitting GDS query: {query_type} ...")

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")

        root = ET.fromstring(body)

        # Look for job ID in the response (GDS uses <JobId>)
        jobid_elem = root.find(".//JobId")
        if jobid_elem is None:
            jobid_elem = root.find(".//Jobid")
        if jobid_elem is None:
            jobid_elem = root.find(".//jobid")
        if jobid_elem is None:
            raise RuntimeError(
                f"GDS async submission failed — no JobId in response:\n{body}"
            )

        return jobid_elem.text.strip()

    @staticmethod
    def _gds_poll_status(job_id, poll_interval=10, max_wait=600):
        """Poll a GDS async job until completion.

        Parameters
        ----------
        job_id : str
            Job ID returned by :meth:`_gds_query_async`.
        poll_interval : int, optional
            Seconds between polls, default 10.
        max_wait : int, optional
            Maximum seconds to wait before raising, default 600.

        Returns
        -------
        list of str
            URLs of result files.
        """
        import time
        import urllib.parse
        import urllib.request
        import xml.etree.ElementTree as ET

        url = f"{Altimetry.GDS_BASE_URL}?{urllib.parse.urlencode({'query': 'status', 'jobid': job_id})}"
        elapsed = 0
        while elapsed < max_wait:
            with urllib.request.urlopen(url, timeout=60) as resp:
                body = resp.read().decode("utf-8")

            root = ET.fromstring(body)
            # Job state is in <StateSummary><State>
            state_elem = root.find(".//StateSummary/State")
            if state_elem is None:
                state_elem = root.find(".//State")
            if state_elem is None:
                state_elem = root.find(".//status")
            status = state_elem.text.strip() if state_elem is not None else "Unknown"

            if status == "Finished":
                # Extract result file URLs
                urls = []
                for elem in root.iter():
                    if elem.text and elem.text.strip().startswith("http"):
                        urls.append(elem.text.strip())
                if not urls:
                    raise RuntimeError(
                        f"GDS job finished but no result URLs found:\n{body}"
                    )
                print(f"GDS query finished. {len(urls)} result file(s) available.")
                return urls

            if status == "Error":
                raise RuntimeError(f"GDS query error:\n{body}")

            print(f"  GDS job status: {status} (elapsed {elapsed}s)")
            time.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"GDS query timed out after {max_wait}s (job_id={job_id})")

    @staticmethod
    def _gds_download_csv(url, save_path):
        """Download a GDS result file.

        Parameters
        ----------
        url : str
            URL of the result file.
        save_path : str
            Local path to save the file.
        """
        import urllib.request

        print(f"Downloading GDS result to {save_path} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"  Download complete: {os.path.getsize(save_path)} bytes")

    def _resolve_time_range(self, scene_date=None, time_buffer_days=365):
        """
        Resolve the t0/t1 time range for SlideRule API requests.

        Uses a three-tier cascade:
        1. Explicit ``scene_date`` parameter
        2. Auto-detect from stereopair XML metadata
        3. Fallback to most recent 2 years

        Parameters
        ----------
        scene_date : str or datetime-like, optional
            Explicit scene date. Parsed via ``pd.Timestamp``.
        time_buffer_days : int, optional
            Days before/after the resolved date, default 365.

        Returns
        -------
        tuple of (str, str, datetime or None)
            (t0_str, t1_str, resolved_date) formatted as
            ``"%Y-%m-%dT%H:%M:%SZ"``. resolved_date is None when
            the 2-year fallback is used.
        """
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        resolved_date = None

        # Tier 1: explicit date
        if scene_date is not None:
            resolved_date = pd.Timestamp(scene_date, tz="UTC").to_pydatetime()

        # Tier 2: auto-detect from XML metadata
        if resolved_date is None:
            try:
                cdate = StereopairMetadataParser(self.directory).get_pair_dict()[
                    "cdate"
                ]
                resolved_date = pd.Timestamp(cdate, tz="UTC").to_pydatetime()
            except Exception:
                pass

        # Compute buffered range if we have a date
        if resolved_date is not None:
            t0 = resolved_date - timedelta(days=time_buffer_days)
            t1 = resolved_date + timedelta(days=time_buffer_days)
            t0 = max(t0, ICESAT2_MISSION_START)
            # If entire range predates mission, fall through to tier 3
            if t1 >= ICESAT2_MISSION_START:
                self._scene_date = resolved_date
                self._t0 = t0
                self._t1 = t1
                return (t0.strftime(fmt), t1.strftime(fmt), resolved_date)

        # Tier 3: fallback to most recent 2 years
        now = datetime.now(tz=timezone.utc)
        t1 = now
        t0 = max(now - timedelta(days=2 * 365), ICESAT2_MISSION_START)

        self._scene_date = None
        self._t0 = t0
        self._t1 = t1
        return (t0.strftime(fmt), t1.strftime(fmt), None)

    @property
    def _time_range_label(self):
        """Return formatted t0–t1 label for plot titles, or empty string if unset."""
        if hasattr(self, "_t0") and hasattr(self, "_t1"):
            return f"{self._t0.strftime('%Y-%m-%d')} to {self._t1.strftime('%Y-%m-%d')}"
        return ""

    @staticmethod
    def _extract_scalar(x):
        """Extract a scalar from an array-valued cell (ndarray or list)."""
        if isinstance(x, np.ndarray):
            return x.item() if x.size == 1 else (x[0] if x.size > 0 else x)
        if isinstance(x, list):
            return x[0] if x else x
        return x

    def request_atl06sr_multi_processing(
        self,
        processing_levels=["all", "ground", "canopy", "top_of_canopy"],
        res=20,
        len=40,
        ats=20,
        cnt=10,
        maxi=6,
        h_sigma_quantile=1.0,
        save_to_parquet=False,
        filename="atl06sr",
        region=None,
        scene_date=None,
        time_buffer_days=365,
    ):
        """
        Request ICESat-2 ATL06-SR data for multiple processing levels.

        Downloads ATL06-SR data from the SlideRule API for specified
        processing levels (surface types), with options to filter and
        save the results. Each processing level targets different surface
        types like ground, canopy, etc.

        Parameters
        ----------
        processing_levels : list, optional
            List of processing levels to request, default is
            ["all", "ground", "canopy", "top_of_canopy"]
        res : int, optional
            ATL06-SR segment resolution in meters, default is 20
        len : int, optional
            ATL06-SR segment length in meters, default is 40
        ats : int, optional
            Along-track sigma, default is 20
        cnt : int, optional
            Minimum number of photons for segment, default is 10
        maxi : int, optional
            Maximum iterations for surface fit, default is 6
        h_sigma_quantile : float, optional
            Quantile for filtering by h_sigma, default is 1.0
        save_to_parquet : bool, optional
            Whether to save results to parquet files, default is False
        filename : str, optional
            Base filename for saved data, default is "atl06sr"
        region : list or None, optional
            Region bounds as [minx, miny, maxx, maxy] in lat/lon,
            default is None (derived from DEM)
        scene_date : str or datetime-like, optional
            Scene acquisition date for server-side time filtering.
            If None, auto-detected from stereopair XML metadata.
            Falls back to most recent 3 years if unavailable.
        time_buffer_days : int, optional
            Days before/after scene_date defining the time window,
            default is 365

        Returns
        -------
        None
            Results are stored in the class attributes

        Notes
        -----
        This method makes SlideRule API calls which require internet connectivity.
        It also samples ESA WorldCover data for land cover classification of the
        ICESat-2 data points. The method includes filtering to improve data quality
        by removing points with high uncertainty and from early mission cycles.
        """
        self._ensure_sliderule()

        if not region:
            region = Raster(self.dem_fn).get_bounds(latlon=True)

        # Resolve server-side time range to limit granules processed
        t0_str, t1_str, resolved_date = self._resolve_time_range(
            scene_date=scene_date, time_buffer_days=time_buffer_days
        )
        if resolved_date is not None:
            print(
                f"Time filter: {t0_str} to {t1_str} "
                f"(+/- {time_buffer_days} days from {resolved_date.date()})"
            )
        else:
            print(f"Time filter: {t0_str} to {t1_str} (2-year fallback)")

        # See parameter discussion on: https://github.com/SlideRuleEarth/sliderule/issues/448
        # "srt": -1 tells the server side code to look at the ATL03 confidence array for each photon
        # and choose the confidence level that is highest across all five surface type entries.
        # cnf options: {"atl03_tep", "atl03_not_considered", "atl03_background", "atl03_within_10m", \
        # "atl03_low", "atl03_medium", "atl03_high"}
        # Note reduced count for limited number of ground photons

        # TODO: use the WorldCover values to determine if we should report canopy or top of canopy
        #   This is tricky, and not clear yet how to go about this. For now, just request all processing levels.

        # TODO: Use more generic variable names and strings for functions that are not just limited to atl06
        #   This can be done when we implement additional requests for other data types

        # Shared parameters for all processing levels
        shared_parms = {
            "poly": region,
            "t0": t0_str,
            "t1": t1_str,
            "res": res,
            "len": len,
            "ats": ats,
            "fit": {"maxi": maxi},
            "samples": {
                "esa_worldcover": {
                    "asset": "esa-worldcover-10meter",
                }
            },
        }

        # Custom parameters for each processing level
        custom_parms = {
            "all": {
                "cnf": "atl03_high",
                "srt": -1,
                "cnt": cnt,
            },
            "ground": {
                "cnf": "atl03_low",
                "srt": -1,
                "cnt": 5,
                "atl08_class": "atl08_ground",
            },
            "canopy": {
                "cnf": "atl03_medium",
                "srt": -1,
                "cnt": 5,
                "atl08_class": "atl08_canopy",
            },
            "top_of_canopy": {
                "cnf": "atl03_medium",
                "srt": -1,
                "cnt": 5,
                "atl08_class": "atl08_top_of_canopy",
            },
        }

        # Filter custom_parms to only include requested processing levels
        custom_parms = {
            key: parms
            for key, parms in custom_parms.items()
            if key in processing_levels
        }

        for key, custom_parm in custom_parms.items():
            parms = {**shared_parms, **custom_parm}

            fn_base = f"{filename}_{key}"

            print(f"\nICESat-2 ATL06 request processing for: {key}")
            fn = f"{fn_base}.parquet"

            print(parms)

            # Check for existing file with matching parameters
            if os.path.exists(fn):
                print(f"Existing file found, reading in: {fn}")
                atl06sr = gpd.read_parquet(fn)

                # Check for parameters in the column
                if "sliderule_parameters" in atl06sr.columns:
                    try:
                        file_parms = json.loads(atl06sr["sliderule_parameters"].iloc[0])
                        parms_copy = parms.copy()
                        parms_copy["poly"] = str(parms_copy["poly"])

                        if str(parms_copy) != str(file_parms):
                            print("Parameters don't match request. Regenerating...")
                            atl06sr = sliderule_api.run("atl03x", parms)
                            if save_to_parquet:
                                self._save_to_parquet(fn, atl06sr, parms)
                    except Exception as e:
                        print(f"Error checking sliderule_parameters column: {e}")
                else:
                    print("No parameters column found, regenerating...")
                    atl06sr = sliderule_api.run("atl03x", parms)
                    if save_to_parquet:
                        self._save_to_parquet(fn, atl06sr, parms)
            else:
                atl06sr = sliderule_api.run("atl03x", parms)
                if save_to_parquet:
                    self._save_to_parquet(fn, atl06sr, parms)

            # Normalize index: x-series returns time_ns (Unix nanoseconds),
            # legacy returns time (GPS seconds). Ensure a DatetimeIndex named "time".
            if atl06sr.index.name == "time_ns" or not isinstance(
                atl06sr.index, pd.DatetimeIndex
            ):
                if "time_ns" in atl06sr.columns:
                    atl06sr.index = pd.to_datetime(atl06sr["time_ns"], unit="ns")
                elif atl06sr.index.name == "time_ns":
                    atl06sr.index = pd.to_datetime(atl06sr.index, unit="ns")
                atl06sr.index.name = "time"

            # Normalize sample columns: x-series may return array values
            # instead of scalars for raster samples (e.g., esa_worldcover.value).
            # Extract the first element from any array-valued cells.
            # After parquet round-trip, arrays may deserialize as lists.
            for col in atl06sr.columns:
                if atl06sr[col].dtype == object and atl06sr.shape[0] > 0:
                    first_val = atl06sr[col].iloc[0]
                    if isinstance(first_val, (np.ndarray, list)):
                        atl06sr[col] = atl06sr[col].apply(self._extract_scalar)

            self.atl06sr_processing_levels[key] = atl06sr

            print(f"Filtering ATL06-SR {key}")

            # From Aimee Gibbons:
            # I'd recommend anything cycle 03 and later, due to pointing issues before cycle 03.
            atl06sr_filtered = atl06sr[atl06sr["cycle"] >= 3]

            # Remove bad fits using high percentile of `h_sigma`, the error estimate for the least squares fit model.
            # TODO: not sure about h_sigma quantile...might throw out too much. Maybe just remove 0 values?
            atl06sr_filtered = atl06sr_filtered[
                atl06sr_filtered["h_sigma"]
                < atl06sr_filtered["h_sigma"].quantile(h_sigma_quantile)
            ]
            # Also need to filter out 0 values, not sure what these are caused by, but also very bad points.
            atl06sr_filtered = atl06sr_filtered[atl06sr_filtered["h_sigma"] != 0]

            self.atl06sr_processing_levels_filtered[key] = atl06sr_filtered

    def _save_to_parquet(self, fn, df, parms):
        """
        Save SlideRule dataframe to parquet including SlideRule parameters.

        Parameters
        ----------
        fn : str
            Filename to save to
        df : pandas.DataFrame
            DataFrame to save
        parms : dict
            SlideRule parameters used to generate the data

        Notes
        -----
        Creates a copy of parameters with polygon coordinates converted to string
        for JSON serialization, and stores these in a column of the DataFrame.
        """
        # We could save the parameters to the parquet metadata, but this
        # was proving rather difficult.
        parms_copy = parms.copy()
        parms_copy["poly"] = str(parms_copy["poly"])
        df["sliderule_parameters"] = json.dumps(parms_copy)
        df.to_parquet(fn)

    def filter_esa_worldcover(self, filter_out="water", retain_only=None):
        """
        Filter ATL06-SR data based on ESA WorldCover land cover classes.

        Filters the data points based on ESA WorldCover classification,
        either by removing specific land cover types or retaining only
        specific types.

        Parameters
        ----------
        filter_out : str, optional
            Land cover group to filter out, default is "water".
            Options: "water", "snow_ice", "trees", "low_vegetation", "built_up"
        retain_only : str or None, optional
            If specified, retain only points matching this land cover group,
            default is None. Same options as ``filter_out``.

        Returns
        -------
        None
            Results are stored in the class attributes

        Notes
        -----
        This method uses the ESA WorldCover land cover classification
        (see ``WORLDCOVER_NAMES``), which was sampled when requesting the
        ATL06-SR data.
        """
        # Groups of WORLDCOVER_NAMES codes for convenient filtering
        value_dict = {
            "water": [80],
            "snow_ice": [70],
            "trees": [10],
            "low_vegetation": [20, 30, 40, 90, 95, 100],
            "built_up": [50],
        }

        if retain_only is not None:
            if retain_only in value_dict:
                values_to_keep = value_dict[retain_only]
                for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
                    if "esa_worldcover.value" in atl06sr.columns:
                        mask = atl06sr["esa_worldcover.value"].isin(values_to_keep)
                        self.atl06sr_processing_levels_filtered[key] = atl06sr[mask]
            else:
                logger.warning(
                    f"\nESA WorldCover retain value not found: {retain_only}\n"
                )
                return

        elif filter_out in value_dict:
            values_to_filter = value_dict[filter_out]
            for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
                if "esa_worldcover.value" in atl06sr.columns:
                    mask = ~atl06sr["esa_worldcover.value"].isin(values_to_filter)
                    self.atl06sr_processing_levels_filtered[key] = atl06sr[mask]
        else:
            logger.warning(f"\nESA WorldCover filter value not found: {filter_out}\n")
            return

    def predefined_temporal_filter_atl06sr(self, date=None):
        """
        Apply predefined temporal filters to ATL06-SR data.

        Creates multiple temporally filtered versions of the ATL06-SR data
        based on a reference date, including:
        - 15-day window around the date
        - 45-day window around the date
        - 91-day window around the date
        - Seasonal filter (same season as the reference date)

        Parameters
        ----------
        date : datetime or None, optional
            Reference date for filtering. If None, extracts date from
            stereopair metadata, default is None

        Returns
        -------
        None
            Results are stored in the class attributes with new keys
            indicating the temporal filter applied

        Notes
        -----
        This method is particularly useful for DEM validation and alignment,
        as it provides multiple temporal windows to analyze the stability
        of the terrain and to identify optimal temporal windows for alignment.
        """
        if (
            date is None
            and hasattr(self, "_scene_date")
            and self._scene_date is not None
        ):
            date = pd.Timestamp(self._scene_date)
        elif date is None:
            date = StereopairMetadataParser(self.directory).get_pair_dict()["cdate"]
        else:
            # Convert to pandas Timestamp to ensure compatibility with DatetimeIndex operations
            date = pd.Timestamp(date)

        # Ensure tz-naive to match the GeoDataFrame DatetimeIndex
        if hasattr(date, "tzinfo") and date.tzinfo is not None:
            date = date.tz_localize(None)

        original_keys = list(self.atl06sr_processing_levels_filtered.keys())

        for key in original_keys:
            print(
                f"\nFiltering ATL06 with 15 day pad, 45 day, 91 day pad, and seasonal pad around {date} for: {key}"
            )
            atl06sr = self.atl06sr_processing_levels_filtered[key]

            fifteen_day = atl06sr[abs(atl06sr.index - date) <= pd.Timedelta(days=15)]
            fortyfive_day = atl06sr[abs(atl06sr.index - date) <= pd.Timedelta(days=45)]
            ninetyone_day = atl06sr[abs(atl06sr.index - date) <= pd.Timedelta(days=91)]

            image_season = date.strftime("%b")
            if image_season in ["Dec", "Jan", "Feb"]:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Dec", "Jan", "Feb"])
                ]
            elif image_season in ["Mar", "Apr", "May"]:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Mar", "Apr", "May"])
                ]
            elif image_season in ["Jun", "Jul", "Aug"]:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Jun", "Jul", "Aug"])
                ]
            else:
                season_filter = atl06sr[
                    atl06sr.index.strftime("%b").isin(["Sep", "Oct", "Nov"])
                ]

            if not fifteen_day.empty:
                self.atl06sr_processing_levels_filtered[f"{key}_15_day_pad"] = (
                    fifteen_day
                )
            if not fortyfive_day.empty:
                self.atl06sr_processing_levels_filtered[f"{key}_45_day_pad"] = (
                    fortyfive_day
                )
            if not ninetyone_day.empty:
                self.atl06sr_processing_levels_filtered[f"{key}_91_day_pad"] = (
                    ninetyone_day
                )
            if not season_filter.empty:
                self.atl06sr_processing_levels_filtered[f"{key}_seasonal"] = (
                    season_filter
                )

    def generic_temporal_filter_atl06sr(
        self, select_years=None, select_months=None, select_days=None
    ):
        """
        Apply custom temporal filters to ATL06-SR data.

        Filters the data based on specific years, months, or days,
        allowing for custom temporal filtering not covered by the
        predefined filters.

        Parameters
        ----------
        select_years : list or None, optional
            Years to keep (e.g., [2019, 2020]), default is None
        select_months : list or None, optional
            Months to keep (1-12), default is None
        select_days : list or None, optional
            Days of month to keep (1-31), default is None

        Returns
        -------
        None
            Results are filtered in-place in the class attributes

        Notes
        -----
        This method modifies the existing filtered data rather than
        creating new entries with different keys. At least one of
        the filter parameters should be provided for the method
        to have any effect.
        """
        for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
            print(f"\nFiltering ATL06 for: {key}")
            atl06_filtered = atl06sr

            if select_years:
                atl06_filtered = atl06_filtered[
                    atl06_filtered.index.year.isin(select_years)
                ]
            if select_months:
                atl06_filtered = atl06_filtered[
                    atl06_filtered.index.month.isin(select_months)
                ]
            if select_days:
                atl06_filtered = atl06_filtered[
                    atl06_filtered.index.day.isin(select_days)
                ]

            self.atl06sr_processing_levels_filtered[key] = atl06_filtered

    def to_csv_for_pc_align(self, key="ground", filename_prefix="atl06sr_for_pc_align"):
        """
        Export ATL06-SR data to CSV format for use with pc_align.

        Creates a CSV file from the filtered ATL06-SR data that is formatted
        for use with the ASP pc_align tool, containing longitude, latitude,
        and height above datum.

        Parameters
        ----------
        key : str, optional
            Processing level to export, default is "ground"
        filename_prefix : str, optional
            Prefix for the output CSV file, default is "atl06sr_for_pc_align"

        Returns
        -------
        str
            Path to the created CSV file

        Notes
        -----
        The ASP pc_align tool requires input data in a specific format.
        This method converts the ATL06-SR GeoDataFrame to the required
        CSV format with columns for longitude, latitude, and height.
        """
        atl06sr = self.atl06sr_processing_levels_filtered[key].to_crs("EPSG:4326")
        csv_fn = f"{filename_prefix}_{key}.csv"
        df = atl06sr[["geometry", "h_mean"]].copy()
        df["lon"] = df["geometry"].x
        df["lat"] = df["geometry"].y
        df["height_above_datum"] = df["h_mean"]
        df = df[["lon", "lat", "height_above_datum"]]
        df.to_csv(csv_fn, header=True, index=False)
        return csv_fn

    def alignment_report(
        self,
        processing_level="ground",
        minimum_points=500,
        agreement_threshold=0.25,
        write_out_aligned_dem=False,
        min_translation_threshold=0.1,
        key_for_aligned_dem="ground",
    ):
        """
        Generate alignment reports and optionally align the DEM.

        Runs pc_align between the DEM and filtered ATL06-SR data for all
        temporal variations of a given processing level, generates reports
        of the alignment results, and optionally creates an aligned DEM.

        Parameters
        ----------
        processing_level : str, optional
            Base processing level to use, default is "ground"
        minimum_points : int, optional
            Minimum number of points required for alignment, default is 500
        agreement_threshold : float, optional
            Threshold for agreement between different temporal alignments,
            as a fraction of the mean shift, default is 0.25
        write_out_aligned_dem : bool, optional
            Whether to create an aligned DEM, default is False
        min_translation_threshold : float, optional
            Minimum translation magnitude as a fraction of the DEM GSD
            to warrant creating an aligned DEM, default is 0.1
        key_for_aligned_dem : str, optional
            Which temporal filter key to use for alignment if
            write_out_aligned_dem is True, default is "ground"

        Returns
        -------
        None
            Sets self.alignment_report_df and optionally self.aligned_dem_fn

        Notes
        -----
        This method performs both the pc_align operations and analysis of the
        alignment results. It checks for consistency across temporal filters
        and only creates an aligned DEM if the translation is significant enough
        and consistent across temporal windows.
        """
        filtered_keys = [
            key
            for key in self.atl06sr_processing_levels_filtered.keys()
            if key.startswith(processing_level)
        ]

        alignment = Alignment(self.directory, self.dem_fn)

        for key in filtered_keys:
            atl06sr = self.atl06sr_processing_levels_filtered[key]
            if atl06sr.shape[0] < minimum_points:
                print(
                    f"\n{key} has {atl06sr.shape[0]} points, which is less than the suggested {minimum_points} points. Skipping alignment.\n"
                )
                continue

            if not glob_file(
                os.path.join(self.directory, "pc_align"), f"*{key}*transform.txt"
            ):
                csv_fn = self.to_csv_for_pc_align(key=key)

                print(f"\nAligning {key} to DEM with pc_align\n")

                alignment.pc_align_dem_to_atl06sr(
                    atl06sr_csv=csv_fn,
                    output_prefix=f"pc_align/pc_align_{key}",
                )

        report_data = []
        for key in filtered_keys:
            report = alignment.pc_align_report(output_prefix=f"pc_align/pc_align_{key}")
            if report:
                report_data.append({"key": key} | report)
        alignment_report_df = pd.DataFrame(report_data)

        if alignment_report_df.empty:
            logger.warning(
                f"\nNo alignment results for processing_level='{processing_level}'. "
                "Check that the requested processing level was included in "
                "request_atl06sr_multi_processing().\n"
            )
            self.alignment_report_df = alignment_report_df
            return

        gsd = Raster(self.dem_fn).get_gsd()
        if (
            alignment_report_df["translation_magnitude"].mean()
            < min_translation_threshold * gsd
        ):
            write_out_aligned_dem = False
            print(
                f"\nTranslation magnitude is less than {min_translation_threshold*100}% of the DEM GSD. Skipping writing out aligned DEM.\n"
            )

        if write_out_aligned_dem:
            # Calculate ranges and mean for each shift component (North-East-Down)
            north_range = (
                alignment_report_df["north_shift"].max()
                - alignment_report_df["north_shift"].min()
            )
            east_range = (
                alignment_report_df["east_shift"].max()
                - alignment_report_df["east_shift"].min()
            )
            down_range = (
                alignment_report_df["down_shift"].max()
                - alignment_report_df["down_shift"].min()
            )
            north_mean = alignment_report_df["north_shift"].mean()
            east_mean = alignment_report_df["east_shift"].mean()
            down_mean = alignment_report_df["down_shift"].mean()

            # Check if range is more than X% of mean for any component
            if (
                north_range > abs(north_mean * agreement_threshold)
                or east_range > abs(east_mean * agreement_threshold)
                or down_range > abs(down_mean * agreement_threshold)
            ):
                print(
                    f"\nWarning: Translation components vary by more than {agreement_threshold*100}% across temporal filters. The translation applied to the aligned DEM may be inaccurate.\n"
                )
                print(
                    f"North shift range: {north_range:.3f} m (mean: {north_mean:.3f} m)"
                )
                print(f"East shift range: {east_range:.3f} m (mean: {east_mean:.3f} m)")
                print(f"Down shift range: {down_range:.3f} m (mean: {down_mean:.3f} m)")

            self.aligned_dem_fn = alignment.apply_dem_translation(
                output_prefix=f"pc_align/pc_align_{key_for_aligned_dem}",
            )
            print(
                f"\nWrote out {key_for_aligned_dem} aligned DEM to {self.aligned_dem_fn}\n"
            )

        self.alignment_report_df = alignment_report_df

    # ------------------------------------------------------------------ #
    #  Planetary altimetry: LOLA (Moon) and MOLA (Mars) via ODE GDS API  #
    # ------------------------------------------------------------------ #

    def request_lola(
        self,
        region=None,
        channels="ttttt",
        email=None,
        save_to_csv=True,
        poll_interval=10,
        max_wait=600,
    ):
        """Request LOLA RDR altimetry data within the DEM bounds.

        Queries the ODE Granular Data System (GDS) REST API for Lunar
        Orbiter Laser Altimeter (LOLA) data.  Results are stored in
        ``self.planetary_points`` as a GeoDataFrame.

        Parameters
        ----------
        region : dict or None, optional
            Bounding box as ``{westernlon, easternlon, minlat, maxlat}``
            in 0-360 longitude.  If None, derived from the DEM.
        channels : str, optional
            LOLA detector channels to include, default ``"ttttt"`` (all 5).
        email : str or None, optional
            Email for GDS job notification.
        save_to_csv : bool, optional
            Cache the downloaded CSV locally, default True.
        poll_interval : int, optional
            Seconds between status polls, default 10.
        max_wait : int, optional
            Maximum seconds to wait for the query, default 600.
        """
        from asp_plot.utils import get_planetary_bounds

        if region is None:
            region = get_planetary_bounds(self.dem_fn, body="moon")

        # Check for cached CSV
        csv_fn = os.path.join(self.directory, "lola_points.csv")
        if os.path.exists(csv_fn):
            print(f"Loading cached LOLA data from {csv_fn}")
            self._load_lola_csv(csv_fn)
            return

        job_id = self._gds_query_async(
            "lolardr",
            region,
            "u",
            email=email,
            channel=channels,
        )
        urls = self._gds_poll_status(
            job_id, poll_interval=poll_interval, max_wait=max_wait
        )

        # Download the CSV — pick the first CSV/text URL
        csv_url = None
        for u in urls:
            if u.endswith(".csv") or u.endswith(".txt") or "result" in u.lower():
                csv_url = u
                break
        if csv_url is None:
            csv_url = urls[0]

        download_path = os.path.join(self.directory, "lola_download.csv")
        self._gds_download_csv(csv_url, download_path)
        self._load_lola_csv(download_path)

        if save_to_csv:
            self.planetary_points.to_csv(csv_fn, index=False)
            print(f"Saved LOLA points to {csv_fn}")

    def _load_lola_csv(self, csv_path):
        """Parse a LOLA simple-topography CSV into a GeoDataFrame.

        The ``results=u`` format returns three columns:
        ``Pt_Longitude``, ``Pt_Latitude``, ``Topography``.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        """
        df = pd.read_csv(csv_path)

        # Normalise column names (GDS sometimes uses different capitalisation)
        cols = {c.strip().lower(): c for c in df.columns}
        lon_col = cols.get("pt_longitude") or cols.get("longitude")
        lat_col = cols.get("pt_latitude") or cols.get("latitude")
        topo_col = cols.get("topography") or cols.get("topo")

        if lon_col is None or lat_col is None or topo_col is None:
            # Fallback: assume positional order lon, lat, topo
            lon_col, lat_col, topo_col = df.columns[0], df.columns[1], df.columns[2]

        df = df.rename(columns={lon_col: "lon", lat_col: "lat", topo_col: "height"})

        # Convert 0-360 → -180/180
        df["lon"] = ((df["lon"] + 180) % 360) - 180

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs='GEOGCRS["Moon",DATUM["D_MOON",ELLIPSOID["MOON",1737400,0]],PRIMEM["Reference_Meridian",0],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]]]',
        )
        self.planetary_points = gdf
        print(f"Loaded {len(gdf)} LOLA points")

    def request_mola(
        self,
        region=None,
        email=None,
        save_to_csv=True,
        poll_interval=10,
        max_wait=600,
    ):
        """Request MOLA PEDR altimetry data within the DEM bounds.

        Queries the ODE GDS REST API for Mars Orbiter Laser Altimeter
        (MOLA) PEDR data.  A -190 m correction is applied to convert
        MOLA radius values (referenced to 3,396,000 m) to heights above
        the IAU sphere (3,396,190 m) used by ASP's ``point2dem``.

        Parameters
        ----------
        region : dict or None, optional
            Bounding box as ``{westernlon, easternlon, minlat, maxlat}``
            in 0-360 longitude.  If None, derived from the DEM.
        email : str or None, optional
            Email for GDS job notification.
        save_to_csv : bool, optional
            Cache the downloaded CSV locally, default True.
        poll_interval : int, optional
            Seconds between status polls, default 10.
        max_wait : int, optional
            Maximum seconds to wait for the query, default 600.
        """
        from asp_plot.utils import get_planetary_bounds

        if region is None:
            region = get_planetary_bounds(self.dem_fn, body="mars")

        # Check for cached CSV
        csv_fn = os.path.join(self.directory, "mola_points.csv")
        if os.path.exists(csv_fn):
            print(f"Loading cached MOLA data from {csv_fn}")
            self._load_mola_csv(csv_fn)
            return

        job_id = self._gds_query_async(
            "molapedr",
            region,
            "v",
            email=email,
        )
        urls = self._gds_poll_status(
            job_id, poll_interval=poll_interval, max_wait=max_wait
        )

        # Download the CSV
        csv_url = None
        for u in urls:
            if u.endswith(".csv") or u.endswith(".txt") or "result" in u.lower():
                csv_url = u
                break
        if csv_url is None:
            csv_url = urls[0]

        download_path = os.path.join(self.directory, "mola_download.csv")
        self._gds_download_csv(csv_url, download_path)
        self._load_mola_csv(download_path)

        if save_to_csv:
            self.planetary_points.to_csv(csv_fn, index=False)
            print(f"Saved MOLA points to {csv_fn}")

    def _load_mola_csv(self, csv_path):
        """Parse a MOLA PEDR CSV into a GeoDataFrame.

        The ``results=v`` format returns a CSV with columns including
        longitude, latitude, and topography/radius values.  A -190 m
        correction is applied if values appear to be MOLA radius
        (referenced to 3,396,000 m sphere) to convert to height above
        the IAU sphere (3,396,190 m) that ASP uses.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        """
        df = pd.read_csv(csv_path)

        # Normalise column names
        cols_lower = {c.strip().lower(): c for c in df.columns}

        # Find lon/lat columns
        lon_col = None
        lat_col = None
        for candidate in ["longitude", "pt_longitude", "areocentric_longitude"]:
            if candidate in cols_lower:
                lon_col = cols_lower[candidate]
                break
        for candidate in ["latitude", "pt_latitude", "areocentric_latitude"]:
            if candidate in cols_lower:
                lat_col = cols_lower[candidate]
                break

        # Find height column — prefer topography, fall back to radius
        height_col = None
        is_radius = False
        for candidate in ["topography", "topo"]:
            if candidate in cols_lower:
                height_col = cols_lower[candidate]
                break
        if height_col is None:
            for candidate in ["radius", "planetary_radius"]:
                if candidate in cols_lower:
                    height_col = cols_lower[candidate]
                    is_radius = True
                    break

        if lon_col is None or lat_col is None or height_col is None:
            # Fallback: use positional
            logger.warning(
                f"Could not identify MOLA CSV columns: {list(df.columns)}. "
                "Using first three columns as lon, lat, height."
            )
            lon_col, lat_col, height_col = df.columns[0], df.columns[1], df.columns[2]

        df = df.rename(
            columns={lon_col: "lon", lat_col: "lat", height_col: "height_raw"}
        )

        # Apply vertical datum correction
        # MOLA radius is relative to 3,396,000 m sphere
        # ASP uses IAU sphere at 3,396,190 m
        # If the value is "MOLA radius" (= planetary_radius - 3,396,000),
        # convert to IAU height (= planetary_radius - 3,396,190) by subtracting 190
        MOLA_SPHERE_OFFSET = 190.0  # meters
        if is_radius:
            df["height"] = df["height_raw"] - MOLA_SPHERE_OFFSET
            print(
                f"Applied -{MOLA_SPHERE_OFFSET} m correction "
                "(MOLA sphere → IAU sphere)"
            )
        else:
            # Topography is already relative to a reference surface.
            # We assume the GDS topography values match ASP's vertical datum
            # well enough for DEM comparison. Document this assumption.
            df["height"] = df["height_raw"]
            logger.info(
                "MOLA topography values used directly. If heights are referenced "
                "to the MOLA areoid rather than the IAU sphere, a systematic offset "
                "may be present in the dh values."
            )

        # Convert 0-360 → -180/180
        df["lon"] = ((df["lon"] + 180) % 360) - 180

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs='GEOGCRS["Mars",DATUM["D_MARS",ELLIPSOID["MARS",3396190,0]],PRIMEM["Reference_Meridian",0],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]]]',
        )
        self.planetary_points = gdf
        print(f"Loaded {len(gdf)} MOLA points")

    def planetary_to_dem_dh(self):
        """Compute height differences between planetary altimetry and DEM.

        Reprojects ``self.planetary_points`` to the DEM CRS, interpolates
        DEM heights at altimetry locations, and computes the difference
        ``altimetry_minus_dem = height - dem_height``.

        The results are stored as new columns on ``self.planetary_points``.
        """
        if self.planetary_points is None or self.planetary_points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return

        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        dem_crs = dem.rio.crs

        # Reproject points to the DEM CRS (use CRS object, not EPSG)
        pts = self.planetary_points.to_crs(dem_crs)

        x = xr.DataArray(pts.geometry.x.values, dims="z")
        y = xr.DataArray(pts.geometry.y.values, dims="z")
        sample = dem.interp(x=x, y=y)

        pts["dem_height"] = sample.values
        pts["altimetry_minus_dem"] = pts["height"] - pts["dem_height"]

        # Update geometry back to geographic CRS for storage
        self.planetary_points = pts.to_crs(self.planetary_points.crs)
        self.planetary_points["dem_height"] = pts["dem_height"].values
        self.planetary_points["altimetry_minus_dem"] = pts["altimetry_minus_dem"].values

        valid = self.planetary_points["altimetry_minus_dem"].dropna()
        print(f"Computed dh for {len(valid)} of {len(self.planetary_points)} points")

    def mapview_plot_planetary_to_dem(
        self,
        clim=None,
        save_dir=None,
        fig_fn=None,
        title=None,
    ):
        """Map view of planetary altimetry vs DEM height differences.

        Plots the DEM hillshade as background with altimetry dh points
        overlaid using a divergent colourmap.

        Parameters
        ----------
        clim : tuple or None, optional
            Colour limits ``(min, max)`` for dh. Default auto.
        save_dir : str or None, optional
            Directory to save figure.
        fig_fn : str or None, optional
            Filename for saved figure.
        title : str or None, optional
            Custom plot title. Auto-detected if None.
        """
        from asp_plot.utils import Raster, detect_planetary_body

        if self.planetary_points is None or self.planetary_points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return

        if "altimetry_minus_dem" not in self.planetary_points.columns:
            self.planetary_to_dem_dh()

        gdf = self.planetary_points.dropna(subset=["altimetry_minus_dem"])
        if gdf.empty:
            logger.warning("No valid dh values for map view.")
            return

        dh = gdf["altimetry_minus_dem"]
        n = len(dh)
        med = np.nanmedian(dh.values)
        nmad = _nmad(dh.values)

        body = detect_planetary_body(self.dem_fn)
        instrument = {"moon": "LOLA", "mars": "MOLA"}.get(body, "Altimetry")
        if title is None:
            title = f"{instrument} vs DEM"

        # Generate hillshade
        dem_raster = Raster(self.dem_fn, downsample=4)
        hs = dem_raster.hillshade()
        extent = rioplot.plotting_extent(dem_raster.ds, transform=dem_raster.transform)

        # Reproject points to DEM CRS for plotting
        dem_crs = dem_raster.ds.crs
        gdf_proj = gdf.to_crs(dem_crs)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=220)
        ax.imshow(hs, cmap="gray", extent=extent, alpha=0.7, interpolation="none")

        # Colour limits
        if clim is None:
            abs_max = max(abs(dh.quantile(0.02)), abs(dh.quantile(0.98)))
            clim = (-abs_max, abs_max)

        gdf_proj.plot(
            ax=ax,
            column="altimetry_minus_dem",
            cmap="RdBu",
            vmin=clim[0],
            vmax=clim[1],
            markersize=2,
            legend=True,
            legend_kwds={"label": f"{instrument} - DEM (m)"},
        )

        stats_text = f"n={n}\nMedian={med:+.2f} m\nNMAD={nmad:.2f} m"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
        )

        ax.set_xticks([])
        ax.set_yticks([])
        fig.suptitle(f"{title}\n(n={n})", size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def histogram_planetary_to_dem(
        self,
        save_dir=None,
        fig_fn=None,
        title=None,
    ):
        """Histogram of planetary altimetry vs DEM height differences.

        Parameters
        ----------
        save_dir : str or None, optional
            Directory to save figure.
        fig_fn : str or None, optional
            Filename for saved figure.
        title : str or None, optional
            Custom plot title. Auto-detected if None.
        """
        from asp_plot.utils import detect_planetary_body

        if self.planetary_points is None or self.planetary_points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return

        if "altimetry_minus_dem" not in self.planetary_points.columns:
            self.planetary_to_dem_dh()

        dh = self.planetary_points["altimetry_minus_dem"].dropna()
        if dh.empty:
            logger.warning("No valid dh values for histogram.")
            return

        n = len(dh)
        med = np.nanmedian(dh.values)
        nmad = _nmad(dh.values)

        body = detect_planetary_body(self.dem_fn)
        instrument = {"moon": "LOLA", "mars": "MOLA"}.get(body, "Altimetry")
        if title is None:
            title = f"{instrument} vs ASP DEM"

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=220)

        xmin = dh.quantile(0.01)
        xmax = dh.quantile(0.99)
        ax.hist(dh.values, bins=128, range=(xmin, xmax), alpha=0.7, color="steelblue")

        stats_text = f"n={n}\nMedian={med:+.2f} m\nNMAD={nmad:.2f} m"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
        )

        ax.set_xlabel(f"{instrument} - DEM (m)")
        ax.set_ylabel("Count")
        fig.suptitle(f"{title}\n(n={n})", size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_atl06sr_time_stamps(
        self,
        key="all",
        title="ICESat-2 ATL06-SR Time Stamps",
        cmap="inferno",
        map_crs="EPSG:4326",
        figsize=(15, 10),
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        """
        Plot ATL06-SR data for different temporal filters.

        Creates a 2x2 grid of plots showing ATL06-SR data for different
        temporal filters (unfiltered, 15-day, 45-day, and seasonal)
        colored by height.

        Parameters
        ----------
        key : str, optional
            Base processing level to plot, default is "all"
        title : str, optional
            Plot title, default is "ICESat-2 ATL06-SR Time Stamps"
        cmap : str, optional
            Matplotlib colormap for elevation, default is "inferno"
        map_crs : str, optional
            Coordinate reference system for mapping, default is "EPSG:4326"
        figsize : tuple, optional
            Figure size as (width, height), default is (15, 10)
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        **ctx_kwargs : dict, optional
            Additional arguments for contextily basemap

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        This method requires the filtered data to have been created using
        the predefined_temporal_filter_atl06sr method for the temporal
        variations to be available.
        """
        time_stamps = ["", "_15_day_pad", "_45_day_pad", "_seasonal"]

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        x_bounds = []
        y_bounds = []
        for ax, time_stamp in zip(axes, time_stamps):
            key_to_plot = f"{key}{time_stamp}"

            if key_to_plot not in self.atl06sr_processing_levels_filtered.keys():
                ax.text(
                    0.5,
                    0.5,
                    f"No points found for {key_to_plot}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            atl06sr = self.atl06sr_processing_levels_filtered[key_to_plot]
            if map_crs:
                crs = map_crs
                ctx_kwargs["crs"] = map_crs
            elif ctx_kwargs:
                crs = ctx_kwargs["crs"]
            else:
                crs = "EPSG:4326"
            atl06sr_sorted = atl06sr.sort_values(by="h_mean").to_crs(crs)
            bounds = atl06sr_sorted.total_bounds
            x_bounds.extend([bounds[0], bounds[2]])
            y_bounds.extend([bounds[1], bounds[3]])

            cb = ColorBar(perc_range=(2, 98))
            cb.get_clim(atl06sr_sorted["h_mean"])
            norm = cb.get_norm(lognorm=False)

            atl06sr_sorted.plot(
                ax=ax,
                column="h_mean",
                cmap=cmap,
                norm=norm,
                s=1,
                legend=True,
                legend_kwds={"label": "Height above datum (m)"},
            )

            ax.set_title(f"{key_to_plot} (n={atl06sr.shape[0]})", size=12)

        # 5% padding
        padding = 0.05
        x_range = max(x_bounds) - min(x_bounds)
        y_range = max(y_bounds) - min(y_bounds)
        for ax, time_stamp in zip(axes, time_stamps):
            key_to_plot = f"{key}{time_stamp}"

            if key_to_plot not in self.atl06sr_processing_levels_filtered.keys():
                continue

            ax.set_xlim(
                min(x_bounds) - padding * x_range, max(x_bounds) + padding * x_range
            )
            ax.set_ylim(
                min(y_bounds) - padding * y_range, max(y_bounds) + padding * y_range
            )
            if ctx_kwargs:
                ctx.add_basemap(ax=ax, **ctx_kwargs)

        suptitle = f"{title}"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=14)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_atl06sr(
        self,
        key="all",
        plot_beams=False,
        plot_dem=False,
        column_name="h_mean",
        cbar_label="Height above datum (m)",
        title="ICESat-2 ATL06-SR",
        clim=None,
        symm_clim=False,
        cmap="inferno",
        map_crs="EPSG:4326",
        figsize=(6, 4),
        save_dir=None,
        fig_fn=None,
        **ctx_kwargs,
    ):
        """
        Plot ATL06-SR data on a map with customizable options.

        Creates a map view of ATL06-SR data with options to color by various
        attributes, highlight different laser beams, overlay on the DEM,
        and add contextual basemaps.

        Parameters
        ----------
        key : str, optional
            Processing level to plot, default is "all"
        plot_beams : bool, optional
            Whether to color points by ICESat-2 beam, default is False
        plot_dem : bool, optional
            Whether to plot the DEM as a background, default is False
        column_name : str, optional
            Column to use for point coloring, default is "h_mean"
        cbar_label : str, optional
            Colorbar label, default is "Height above datum (m)"
        title : str, optional
            Plot title, default is "ICESat-2 ATL06-SR"
        clim : tuple or None, optional
            Color limits as (min, max), default is None (auto)
        symm_clim : bool, optional
            Whether to use symmetric color limits, default is False
        cmap : str, optional
            Matplotlib colormap, default is "inferno"
        map_crs : str, optional
            Coordinate reference system for mapping, default is "EPSG:4326"
        figsize : tuple, optional
            Figure size as (width, height), default is (6, 4)
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        **ctx_kwargs : dict, optional
            Additional arguments for contextily basemap

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        When plot_beams is True, points are colored by ICESat-2 laser spot
        number, with strong beams (1, 3, 5) in darker colors and
        weak beams (2, 4, 6) in lighter colors.
        """
        atl06sr = self.atl06sr_processing_levels_filtered[key]
        atl06sr_sorted = atl06sr.sort_values(by=column_name).to_crs(map_crs)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=220)

        if plot_dem:
            ctx_kwargs = {}
            # We downsample to speed plotting. This is not carried over into any analysis.
            dem_downsampled = Raster(self.dem_fn, downsample=10)
            cb = ColorBar(perc_range=(2, 98))
            cb.get_clim(dem_downsampled.data)
            # Plot using rasterio's show function
            rioplot.show(
                dem_downsampled.data,
                transform=dem_downsampled.transform,
                ax=ax,
                cmap="inferno",
                vmin=cb.clim[0],
                vmax=cb.clim[1],
                alpha=1,
            )
            ax.set_title(None)

        # TODO: Implement optional hillshade plotting

        if plot_beams:
            color_dict = {
                1: "red",
                2: "lightpink",
                3: "blue",
                4: "lightblue",
                5: "green",
                6: "lightgreen",
            }
            patches = [mpatches.Patch(color=v, label=k) for k, v in color_dict.items()]
            atl06sr_sorted.plot(
                ax=ax,
                markersize=1,
                color=atl06sr_sorted["spot"].map(color_dict).values,
            )
            ax.legend(
                handles=patches, title="laser spot\n(strong=1,3,5)", loc="upper left"
            )
        else:
            if plot_dem:
                cb.symm = symm_clim
            else:
                cb = ColorBar(perc_range=(2, 98), symm=symm_clim)
                if clim is None:
                    cb.get_clim(atl06sr_sorted[column_name])
                else:
                    cb.clim = clim

            norm = cb.get_norm(lognorm=False)

            atl06sr_sorted.plot(
                ax=ax,
                column=column_name,
                cmap=cmap,
                norm=norm,
                s=1,
                legend=True,
                legend_kwds={"label": cbar_label},
            )

        if ctx_kwargs:
            ctx.add_basemap(ax=ax, **ctx_kwargs)

        ax.set_xticks([])
        ax.set_yticks([])

        suptitle = f"{title}\n{key} (n={atl06sr.shape[0]})"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def atl06sr_to_dem_dh(self):
        """
        Calculate height differences between ATL06-SR data and DEMs.

        Interpolates DEM heights at ATL06-SR point locations and calculates
        the difference between ICESat-2 heights and DEM heights. If an aligned
        DEM is available, also calculates differences against it.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Adds columns to the filtered ATL06-SR data:
            - dem_height: Interpolated height from the original DEM
            - icesat_minus_dem: Height difference (ICESat-2 - DEM)
            - aligned_dem_height: Interpolated height from aligned DEM (if available)
            - icesat_minus_aligned_dem: Height difference with aligned DEM (if available)

        Notes
        -----
        This method performs bilinear interpolation of DEM values at
        ATL06-SR point locations using xarray and rioxarray. It handles
        coordinate system conversions automatically.
        """
        dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
        epsg = dem.rio.crs.to_epsg()
        for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
            atl06sr = atl06sr.to_crs(f"EPSG:{epsg}")

            x = xr.DataArray(atl06sr.geometry.x.values, dims="z")
            y = xr.DataArray(atl06sr.geometry.y.values, dims="z")
            sample = dem.interp(x=x, y=y)

            atl06sr["dem_height"] = sample.values
            atl06sr["icesat_minus_dem"] = atl06sr["h_mean"] - atl06sr["dem_height"]
            self.atl06sr_processing_levels_filtered[key] = atl06sr

        if self.aligned_dem_fn:
            aligned_dem = rioxarray.open_rasterio(
                self.aligned_dem_fn, masked=True
            ).squeeze()
            epsg = aligned_dem.rio.crs.to_epsg()
            for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
                atl06sr = atl06sr.to_crs(f"EPSG:{epsg}")

                x = xr.DataArray(atl06sr.geometry.x.values, dims="z")
                y = xr.DataArray(atl06sr.geometry.y.values, dims="z")
                sample = aligned_dem.interp(x=x, y=y)

                atl06sr["aligned_dem_height"] = sample.values
                atl06sr["icesat_minus_aligned_dem"] = (
                    atl06sr["h_mean"] - atl06sr["aligned_dem_height"]
                )
                self.atl06sr_processing_levels_filtered[key] = atl06sr

    def mapview_plot_atl06sr_to_dem(
        self,
        key="all",
        clim=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
        map_crs=None,
        **ctx_kwargs,
    ):
        """
        Plot height differences between ATL06-SR data and DEMs.

        Creates a map visualization of the height differences between
        ICESat-2 ATL06-SR data and either the original or aligned DEM.

        Parameters
        ----------
        key : str, optional
            Processing level to plot, default is "all"
        clim : tuple or None, optional
            Color limits as (min, max), default is None (auto)
        plot_aligned : bool, optional
            Whether to plot differences with aligned DEM, default is False
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        map_crs : str or None, optional
            Coordinate reference system for mapping, default is None
            (use DEM's CRS)
        **ctx_kwargs : dict, optional
            Additional arguments for contextily basemap

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        If the height differences haven't been calculated yet,
        this method calls atl06sr_to_dem_dh() to calculate them.
        The plot uses a divergent colormap (RdBu) to highlight
        positive and negative differences.
        """
        if plot_aligned:
            column_name = "icesat_minus_aligned_dem"
            if not self.aligned_dem_fn:
                print("\nAligned DEM not found.\n")
                return
        else:
            column_name = "icesat_minus_dem"

        atl06sr = self.atl06sr_processing_levels_filtered[key]

        if column_name not in atl06sr.columns:
            print(
                f"\n{column_name} not found in ATL06 dataframe: {key}. Running differencing first.\n"
            )
            self.atl06sr_to_dem_dh()

        if clim is not None:
            symm_clim = False
        else:
            symm_clim = True

        if not map_crs:
            dem = rioxarray.open_rasterio(self.dem_fn, masked=True).squeeze()
            epsg = dem.rio.crs.to_epsg()
            map_crs = f"EPSG:{epsg}"

        self.plot_atl06sr(
            key=key,
            column_name=column_name,
            cbar_label="ICESat-2 minus DEM (m)",
            clim=clim,
            symm_clim=symm_clim,
            cmap="RdBu",
            map_crs=map_crs,
            save_dir=save_dir,
            fig_fn=fig_fn,
            **ctx_kwargs,
        )

    def histogram(
        self,
        key="all",
        title="Histogram",
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot histograms of height differences between ATL06-SR data and DEMs.

        Creates histograms of the height differences between ICESat-2 ATL06-SR
        data and DEMs, with statistics including median and normalized median
        absolute deviation (NMAD).

        Parameters
        ----------
        key : str, optional
            Processing level to plot, default is "all"
        title : str, optional
            Plot title, default is "Histogram"
        plot_aligned : bool, optional
            Whether to include differences with aligned DEM, default is False
        save_dir : str or None, optional
            Directory to save figure, default is None (don't save)
        fig_fn : str or None, optional
            Filename for saved figure, default is None

        Returns
        -------
        None
            Displays the plot and optionally saves it

        Notes
        -----
        If the height differences haven't been calculated yet,
        this method calls atl06sr_to_dem_dh() to calculate them.
        NMAD is a robust measure of dispersion that is less sensitive
        to outliers than standard deviation, calculated as
        1.4826 * median(abs(x - median(x))).
        """

        atl06sr = self.atl06sr_processing_levels_filtered[key]

        if "icesat_minus_dem" not in atl06sr.columns:
            print(
                f"\n'icesat_minus_dem' not found in ATL06 dataframe: {key}. Running differencing first.\n"
            )
            self.atl06sr_to_dem_dh()
            atl06sr = self.atl06sr_processing_levels_filtered[key]

        column_names = ["icesat_minus_dem"]
        if plot_aligned:
            column_names.append("icesat_minus_aligned_dem")
            if not self.aligned_dem_fn:
                print("\nAligned DEM not found.\n")
                return

        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=220)

        for column_name in column_names:
            med = atl06sr[column_name].quantile(0.50)
            nmad = atl06sr[[column_name]].apply(_nmad).iloc[0]

            xmin = atl06sr[column_name].quantile(0.01)
            xmax = atl06sr[column_name].quantile(0.99)
            plot_kwargs = {"bins": 128, "alpha": 0.5, "range": (xmin, xmax)}
            atl06sr.hist(
                ax=ax,
                column=column_name,
                label=f"{column_name}, Median={med:0.2f}, NMAD={nmad:0.2f}",
                **plot_kwargs,
            )

        ax.legend()
        ax.set_title(None)
        ax.set_xlabel("ICESat-2 - DEM (m)")

        suptitle = f"{title}\n{key} (n={atl06sr.shape[0]})"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=10)

        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def _select_best_track(self, key="all"):
        """
        Select the RGT/cycle/spot combination with the most valid ATL06-SR points.

        Parameters
        ----------
        key : str, optional
            Processing level key, default is "all"

        Returns
        -------
        dict or None
            Dictionary with keys: rgt, cycle, spot, count, date.
            Returns None if no valid tracks found.
        """
        atl06sr = self.atl06sr_processing_levels_filtered[key]

        if "icesat_minus_dem" not in atl06sr.columns:
            self.atl06sr_to_dem_dh()
            atl06sr = self.atl06sr_processing_levels_filtered[key]

        valid = atl06sr.dropna(subset=["icesat_minus_dem"])
        if valid.empty:
            return None

        pass_counts = valid.groupby(["rgt", "cycle", "spot"]).size()
        best = pass_counts.idxmax()
        best_data = valid[
            (valid["rgt"] == best[0])
            & (valid["cycle"] == best[1])
            & (valid["spot"] == best[2])
        ]

        date_str = best_data.index[0].strftime("%Y-%m-%d")

        return {
            "rgt": best[0],
            "cycle": best[1],
            "spot": best[2],
            "count": int(pass_counts.loc[best]),
            "date": date_str,
        }

    def histogram_by_landcover(
        self,
        key="all",
        top_n=4,
        title="ICESat-2 ATL06-SR vs DEM",
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot histogram of dh with per-landcover-class statistics.

        Creates a histogram of the height differences between ICESat-2
        ATL06-SR data and the DEM, with a text annotation showing overall
        and per-landcover-class statistics (count, median, NMAD).

        Parameters
        ----------
        key : str, optional
            Processing level key, default is "all"
        top_n : int, optional
            Number of top landcover classes to report, default is 4
        title : str, optional
            Plot title, default is "ICESat-2 ATL06-SR vs DEM"
        save_dir : str or None, optional
            Directory to save figure, default is None
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        """
        atl06sr = self.atl06sr_processing_levels_filtered[key]

        if "icesat_minus_dem" not in atl06sr.columns:
            self.atl06sr_to_dem_dh()
            atl06sr = self.atl06sr_processing_levels_filtered[key]

        dh = atl06sr["icesat_minus_dem"].dropna()
        if dh.empty:
            logger.warning(f"\nNo valid dh values for key: {key}\n")
            return

        overall_med = np.nanmedian(dh.values)
        overall_nmad = _nmad(dh.values)
        overall_n = len(dh)

        stats_lines = [
            f"All: n={overall_n}, Med={overall_med:+.2f} m, NMAD={overall_nmad:.2f} m"
        ]

        wc_col = "esa_worldcover.value"
        if wc_col in atl06sr.columns:
            valid = atl06sr.dropna(subset=["icesat_minus_dem"])
            valid_wc = valid.dropna(subset=[wc_col])

            if not valid_wc.empty:
                valid_wc = valid_wc.copy()
                valid_wc["lc_name"] = valid_wc[wc_col].map(WORLDCOVER_NAMES)
                valid_wc["lc_name"] = valid_wc["lc_name"].fillna("Unknown")

                grouped = valid_wc.groupby("lc_name")["icesat_minus_dem"]
                class_stats = []
                for name, group in grouped:
                    if len(group) >= 10:
                        class_stats.append(
                            {
                                "name": name,
                                "n": len(group),
                                "med": np.nanmedian(group.values),
                                "nmad": _nmad(group.values),
                            }
                        )

                class_stats.sort(key=lambda x: x["n"], reverse=True)
                class_stats = class_stats[:top_n]

                if class_stats:
                    stats_lines.append("─" * 35)
                    for cs in class_stats:
                        stats_lines.append(
                            f"{cs['name']}: n={cs['n']}, Med={cs['med']:+.2f}, NMAD={cs['nmad']:.2f}"
                        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=220)

        xmin = dh.quantile(0.01)
        xmax = dh.quantile(0.99)
        ax.hist(dh.values, bins=128, range=(xmin, xmax), alpha=0.7, color="steelblue")

        stats_text = "\n".join(stats_lines)
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9),
        )

        ax.set_xlabel("ICESat-2 - DEM (m)")
        ax.set_ylabel("Count")
        suptitle = f"{title}\n{key} (n={overall_n})"
        if self._time_range_label:
            suptitle += f"\n{self._time_range_label}"
        fig.suptitle(suptitle, size=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)

    def plot_atl06sr_dem_profile(
        self,
        key="all",
        rgt=None,
        cycle=None,
        spot=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        """
        Plot elevation profile comparing ICESat-2 and DEM along the best track.

        Creates a three-row figure:
        - Row 1: Combined elevation profile (left y-axis) and dh (right y-axis)
        - Row 2: Two 1 km zoom segments at the locations of minimum and
          maximum rolling-mean dh, with normalized elevations
        - Row 3: DEM hillshade map with the full track and segment extents

        Parameters
        ----------
        key : str, optional
            Processing level key, default is "all"
        rgt : int or None, optional
            Reference ground track (auto-selected if None)
        cycle : int or None, optional
            Cycle number (auto-selected if None)
        spot : int or None, optional
            Spot number (auto-selected if None)
        plot_aligned : bool, optional
            Whether to also plot the aligned DEM profile, default is False
        save_dir : str or None, optional
            Directory to save figure, default is None
        fig_fn : str or None, optional
            Filename for saved figure, default is None
        """
        if not all([rgt, cycle, spot]):
            best = self._select_best_track(key)
            if best is None:
                logger.warning("\nNo valid tracks found for profile plot. Skipping.\n")
                return
            rgt, cycle, spot = best["rgt"], best["cycle"], best["spot"]
            track_count = best["count"]
            track_date = best["date"]
        else:
            track_count = None
            track_date = None

        atl06sr = self.atl06sr_processing_levels_filtered[key]

        if "icesat_minus_dem" not in atl06sr.columns:
            self.atl06sr_to_dem_dh()
            atl06sr = self.atl06sr_processing_levels_filtered[key]

        track = atl06sr[
            (atl06sr["rgt"] == rgt)
            & (atl06sr["cycle"] == cycle)
            & (atl06sr["spot"] == spot)
        ].copy()

        if track.empty:
            logger.warning(
                f"\nNo data for RGT={rgt}, Cycle={cycle}, Spot={spot}. Skipping profile.\n"
            )
            return

        track = track.sort_values("x_atc")
        x_atc = track["x_atc"].values
        dist = (track["x_atc"] - track["x_atc"].min()) / 1000.0

        if track_date is None:
            track_date = track.index[0].strftime("%Y-%m-%d")
        if track_count is None:
            track_count = len(track)

        # --- Segment selection: sliding 1 km window scored by |med(dh)| + NMAD ---
        dh_col = "icesat_minus_dem"
        dh_vals = track[dh_col].dropna()
        track_length_m = x_atc[-1] - x_atc[0] if len(x_atc) > 1 else 0
        diffs = np.diff(x_atc)
        median_spacing = np.median(diffs) if len(diffs) > 0 else 0
        show_segments = median_spacing > 0 and track_length_m >= 1000

        seg_best_mask = None
        seg_worst_mask = None
        if show_segments:
            # Evaluate candidate segments centered on each point
            half_win = 500  # meters
            scores = []
            for i, xc in enumerate(x_atc):
                mask_i = (track["x_atc"] >= xc - half_win) & (
                    track["x_atc"] <= xc + half_win
                )
                seg_i = track.loc[mask_i]
                n_total = len(seg_i)
                if n_total < 3:
                    scores.append(np.nan)
                    continue
                # Require >75% DEM coverage
                dem_valid = seg_i["dem_height"].notna().sum()
                if dem_valid / n_total < 0.75:
                    scores.append(np.nan)
                    continue
                seg_dh = seg_i[dh_col].dropna().values
                if len(seg_dh) < 3:
                    scores.append(np.nan)
                    continue
                scores.append(abs(np.nanmedian(seg_dh)) + _nmad(seg_dh))

            scores = np.array(scores)
            valid_scores = ~np.isnan(scores)
            if valid_scores.sum() >= 2:
                idx_best = np.nanargmin(scores)
                idx_worst = np.nanargmax(scores)
                x_atc_lo = x_atc[0]
                x_atc_hi = x_atc[-1]

                best_center = x_atc[idx_best]
                worst_center = x_atc[idx_worst]

                seg_best_start = max(best_center - half_win, x_atc_lo)
                seg_best_end = min(best_center + half_win, x_atc_hi)
                seg_worst_start = max(worst_center - half_win, x_atc_lo)
                seg_worst_end = min(worst_center + half_win, x_atc_hi)

                seg_best_mask = (track["x_atc"] >= seg_best_start) & (
                    track["x_atc"] <= seg_best_end
                )
                seg_worst_mask = (track["x_atc"] >= seg_worst_start) & (
                    track["x_atc"] <= seg_worst_end
                )

                # Convert to km for axvspan on Row 1
                seg_best_start_km = (seg_best_start - x_atc[0]) / 1000.0
                seg_best_end_km = (seg_best_end - x_atc[0]) / 1000.0
                seg_worst_start_km = (seg_worst_start - x_atc[0]) / 1000.0
                seg_worst_end_km = (seg_worst_end - x_atc[0]) / 1000.0
            else:
                show_segments = False

        # --- Figure layout ---
        if show_segments:
            fig = plt.figure(figsize=(12, 14))
            gs = fig.add_gridspec(
                3, 2, height_ratios=[2, 1.5, 2], hspace=0.3, wspace=0.25
            )
            ax1 = fig.add_subplot(gs[0, :])
            ax2_left = fig.add_subplot(gs[1, 0])
            ax2_right = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[2, :])
        else:
            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(2, 1, height_ratios=[2, 2], hspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax3 = fig.add_subplot(gs[1])

        # ===================== Row 1: Combined profile =====================
        # Left y-axis: elevation
        valid_dem = track["dem_height"].dropna()
        if not valid_dem.empty:
            ax1.plot(
                dist.loc[valid_dem.index],
                valid_dem,
                color="gray",
                linewidth=1,
                label="DEM (left axis)",
                zorder=1,
            )

        if plot_aligned and self.aligned_dem_fn:
            if "aligned_dem_height" in track.columns:
                valid_aligned = track["aligned_dem_height"].dropna()
                if not valid_aligned.empty:
                    ax1.plot(
                        dist.loc[valid_aligned.index],
                        valid_aligned,
                        color="orange",
                        linewidth=1,
                        label="Aligned DEM (left axis)",
                        zorder=2,
                    )

        ax1.scatter(
            dist,
            track["h_mean"],
            color="steelblue",
            s=8,
            label="ICESat-2 ATL06-SR (left axis)",
            zorder=3,
        )

        ax1.set_ylabel("Elevation (m HAE)")
        ax1.set_xlabel("Along-track distance (km)")

        # Segment highlight spans
        seg_best_color = "tab:blue"
        seg_worst_color = "tab:red"
        if show_segments:
            ax1.axvspan(
                seg_best_start_km,
                seg_best_end_km,
                alpha=0.15,
                color=seg_best_color,
                zorder=0,
            )
            ax1.axvspan(
                seg_worst_start_km,
                seg_worst_end_km,
                alpha=0.15,
                color=seg_worst_color,
                zorder=0,
            )

        # Right y-axis: dh
        ax1_dh = ax1.twinx()
        if not dh_vals.empty:
            med = np.nanmedian(dh_vals.values)
            nmad_val = _nmad(dh_vals.values)
            ax1_dh.scatter(
                dist.loc[dh_vals.index],
                dh_vals,
                color="salmon",
                s=4,
                alpha=0.6,
                zorder=4,
                label=f"dh (right axis) Med={med:+.2f} m, NMAD={nmad_val:.2f} m",
            )
            ax1_dh.axhline(0, color="black", linewidth=0.5, linestyle="--", zorder=1)
            ax1_dh.set_ylabel("ICESat-2 − DEM (m)")

        # Combined legend from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax1_dh.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")

        # =============== Row 2: Zoom segments (if available) ===============
        if show_segments:
            for ax_seg, mask, color, label_prefix in [
                (ax2_left, seg_best_mask, seg_best_color, "Best"),
                (ax2_right, seg_worst_mask, seg_worst_color, "Worst"),
            ]:
                seg = track.loc[mask]

                # DEM line
                seg_dem = seg["dem_height"].dropna()
                seg_h = seg["h_mean"].dropna()
                if not seg_dem.empty:
                    seg_dem_dist = (
                        seg.loc[seg_dem.index, "x_atc"].values - seg["x_atc"].values[0]
                    )
                    ax_seg.plot(
                        seg_dem_dist,
                        seg_dem.values,
                        color="gray",
                        linewidth=1,
                        label="DEM",
                    )
                # ICESat-2 scatter
                if not seg_h.empty:
                    seg_h_dist = (
                        seg.loc[seg_h.index, "x_atc"].values - seg["x_atc"].values[0]
                    )
                    ax_seg.scatter(
                        seg_h_dist,
                        seg_h.values,
                        color="steelblue",
                        s=8,
                        label="ICESat-2",
                    )

                seg_dh = seg[dh_col].dropna()
                seg_med = np.nanmedian(seg_dh.values) if not seg_dh.empty else 0
                seg_nmad = _nmad(seg_dh.values) if len(seg_dh) >= 3 else 0
                ax_seg.set_title(
                    f"{label_prefix} (Med={seg_med:+.1f} m, NMAD={seg_nmad:.1f} m)",
                    fontsize=9,
                    color=color,
                )
                ax_seg.set_xlabel("Along-track distance (m)")
                ax_seg.set_ylabel("Elevation (m HAE)")
                # Add background color tint matching the span color
                ax_seg.set_facecolor((*plt.matplotlib.colors.to_rgb(color), 0.05))

            ax2_left.legend(fontsize=7, loc="best")

        # =================== Row 3: Map view ====================
        try:
            raster = Raster(self.dem_fn, downsample=5)
            dem_data, dem_extent = raster.read_array(extent=True)
            epsg = raster.get_epsg_code()

            # Generate hillshade from downsampled data using matplotlib
            from matplotlib.colors import LightSource

            ls = LightSource(azdeg=315, altdeg=45)
            # Fill masked values for hillshade computation
            fill_val = np.nanmedian(np.asarray(dem_data))
            dem_filled = np.asarray(np.ma.filled(dem_data, fill_val))
            hs = ls.hillshade(dem_filled)

            ax3.imshow(
                hs,
                extent=dem_extent,
                cmap="gray",
                origin="upper",
                aspect="equal",
            )
            ax3.imshow(
                dem_data,
                extent=dem_extent,
                cmap="terrain",
                alpha=0.4,
                origin="upper",
                aspect="equal",
            )

            # Reproject track to DEM CRS and plot
            track_proj = track.to_crs(f"EPSG:{epsg}")
            ax3.plot(
                track_proj.geometry.x,
                track_proj.geometry.y,
                color="black",
                linewidth=2,
                label="Track",
                zorder=5,
            )

            # Overlay segment extents
            if show_segments:
                for mask, color, label in [
                    (seg_best_mask, seg_best_color, "Best"),
                    (seg_worst_mask, seg_worst_color, "Worst"),
                ]:
                    seg_proj = track_proj.loc[mask]
                    if not seg_proj.empty:
                        ax3.plot(
                            seg_proj.geometry.x,
                            seg_proj.geometry.y,
                            color=color,
                            linewidth=4,
                            label=label,
                            zorder=6,
                        )

            # Zoom to track extent with padding
            track_bounds = track_proj.total_bounds  # [minx, miny, maxx, maxy]
            dx = track_bounds[2] - track_bounds[0]
            dy = track_bounds[3] - track_bounds[1]
            pad = max(dx, dy) * 0.2
            ax3.set_xlim(track_bounds[0] - pad, track_bounds[2] + pad)
            ax3.set_ylim(track_bounds[1] - pad, track_bounds[3] + pad)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.legend(fontsize=8, loc="upper left")
        except Exception:
            logger.warning("Could not generate map view for profile plot.")
            ax3.text(
                0.5,
                0.5,
                "Map view unavailable",
                transform=ax3.transAxes,
                ha="center",
                va="center",
            )
            ax3.set_xticks([])
            ax3.set_yticks([])

        # Title
        title_str = f"RGT {rgt}, Cycle {cycle}, Spot {spot} ({track_date})"
        if track_count:
            title_str += f" — n={track_count}"
        if self._time_range_label:
            title_str += f"\n{self._time_range_label}"
        fig.suptitle(title_str, size=10)

        # Use subplots_adjust instead of tight_layout to avoid warnings with twinx
        fig.subplots_adjust(top=0.93, bottom=0.04, left=0.08, right=0.92)
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn)
