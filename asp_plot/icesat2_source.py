"""ICESat-2 ATL06-SR altimetry source.

Owns the Earth/ICESat-2 side of :class:`asp_plot.altimetry.Altimetry`:
requesting ATL06-SR points from SlideRule (with parquet caching), sampling
ESA WorldCover, temporal and outlier filtering, differencing against the ASP
DEM, and selecting the best track / best-worst 1 km segments used by the
profile plots and reproducibility (#121).

The class holds its own data (``atl06sr_processing_levels`` /
``atl06sr_processing_levels_filtered`` and the resolved request time range)
and reads the cross-cutting ``dem_fn`` / ``directory`` / ``aligned_dem_fn``
from the coordinating :class:`Altimetry` instance passed at construction.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
from sliderule import sliderule as sliderule_api

from asp_plot.altimetry_source import AltimetrySource
from asp_plot.stereopair_metadata_parser import StereopairMetadataParser
from asp_plot.utils import Raster
from asp_plot.utils import nmad as _nmad

logger = logging.getLogger(__name__)

ICESAT2_MISSION_START = datetime(2018, 10, 14, tzinfo=timezone.utc)


class Icesat2Source(AltimetrySource):
    """Request, cache, filter and difference ICESat-2 ATL06-SR points.

    Parameters
    ----------
    alt : Altimetry
        The coordinating :class:`asp_plot.altimetry.Altimetry` instance.
        Cross-cutting paths (``dem_fn``, ``directory``, ``aligned_dem_fn``)
        are read from it so a single source of truth describes the DEM
        under analysis.
    atl06sr_processing_levels : dict, optional
        Pre-loaded ATL06-SR data for different processing levels.
    atl06sr_processing_levels_filtered : dict, optional
        Pre-loaded filtered ATL06-SR data.
    """

    def __init__(
        self,
        alt,
        atl06sr_processing_levels=None,
        atl06sr_processing_levels_filtered=None,
    ):
        super().__init__(alt)
        self.atl06sr_processing_levels = (
            {} if atl06sr_processing_levels is None else atl06sr_processing_levels
        )
        self.atl06sr_processing_levels_filtered = (
            {}
            if atl06sr_processing_levels_filtered is None
            else atl06sr_processing_levels_filtered
        )

        # Lazy SlideRule initialization — only needed for ICESat-2 methods
        self._sliderule_initialized = False

    def _ensure_sliderule(self):
        """Initialize the SlideRule session on first use."""
        if not self._sliderule_initialized:
            sliderule_api.init(
                "slideruleearth.io", verbose=False, loglevel=logging.WARNING
            )
            # Also silence the chatty sliderule.session logger
            logging.getLogger("sliderule.session").setLevel(logging.WARNING)
            self._sliderule_initialized = True

    def _resolve_time_range(
        self,
        time_range="all",
        scene_date=None,
        time_buffer_days=365,
        t0=None,
        t1=None,
    ):
        """
        Resolve the t0/t1 time range for SlideRule API requests.

        Parameters
        ----------
        time_range : str, optional
            ``"all"`` (default) returns full ICESat-2 mission range.
            ``"buffered"`` activates the cascade: explicit ``t0``/``t1``
            > ``scene_date`` ± ``time_buffer_days`` > XML metadata
            ± ``time_buffer_days`` > fall back to ``"all"``.
        scene_date : str or datetime-like, optional
            Explicit scene date. Only used when ``time_range="buffered"``.
        time_buffer_days : int, optional
            Days before/after the resolved date, default 365.
        t0 : str or datetime-like, optional
            Explicit start date. Only used when ``time_range="buffered"``.
        t1 : str or datetime-like, optional
            Explicit end date. Defaults to present if only ``t0`` given.

        Returns
        -------
        tuple of (str, str, datetime or None)
            (t0_str, t1_str, resolved_date) formatted as
            ``"%Y-%m-%dT%H:%M:%SZ"``. resolved_date is the scene date
            when buffered from a date, otherwise None.
        """
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        # Truncate to midnight UTC so cached parquet params stay stable within a day
        today = datetime.now(tz=timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        def _set_all():
            self._scene_date = None
            self._t0 = ICESAT2_MISSION_START
            self._t1 = today
            return (
                ICESAT2_MISSION_START.strftime(fmt),
                today.strftime(fmt),
                None,
            )

        if time_range == "all":
            return _set_all()

        # time_range == "buffered": cascade t0/t1 > scene_date > XML > all

        # 1. Explicit t0/t1
        if t0 is not None:
            t0_dt = pd.Timestamp(t0, tz="UTC").to_pydatetime()
            t1_dt = (
                pd.Timestamp(t1, tz="UTC").to_pydatetime() if t1 is not None else today
            )
            self._scene_date = None
            self._t0 = t0_dt
            self._t1 = t1_dt
            return (t0_dt.strftime(fmt), t1_dt.strftime(fmt), None)

        # 2. Explicit scene_date
        resolved_date = None
        if scene_date is not None:
            resolved_date = pd.Timestamp(scene_date, tz="UTC").to_pydatetime()

        # 3. Auto-detect from XML metadata
        if resolved_date is None:
            try:
                cdate = StereopairMetadataParser(self.alt.directory).get_pair_dict()[
                    "cdate"
                ]
                resolved_date = pd.Timestamp(cdate, tz="UTC").to_pydatetime()
            except Exception:
                pass

        # Compute buffered range if we have a date
        if resolved_date is not None:
            t0_dt = resolved_date - timedelta(days=time_buffer_days)
            t1_dt = resolved_date + timedelta(days=time_buffer_days)
            t0_dt = max(t0_dt, ICESAT2_MISSION_START)
            if t1_dt >= ICESAT2_MISSION_START:
                self._scene_date = resolved_date
                self._t0 = t0_dt
                self._t1 = t1_dt
                return (t0_dt.strftime(fmt), t1_dt.strftime(fmt), resolved_date)

        # Fallback: all
        return _set_all()

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
        time_range="all",
        scene_date=None,
        time_buffer_days=365,
        t0=None,
        t1=None,
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
        time_range : str, optional
            ``"all"`` (default) requests all ICESat-2 data from mission
            start to present. ``"buffered"`` activates time filtering
            via the cascade: ``t0``/``t1`` > ``scene_date`` ±
            ``time_buffer_days`` > XML metadata ±
            ``time_buffer_days`` > fall back to all.
        scene_date : str or datetime-like, optional
            Scene acquisition date, used when ``time_range="buffered"``.
            If None, auto-detected from stereopair XML metadata.
        time_buffer_days : int, optional
            Days before/after scene_date defining the time window,
            default is 365
        t0 : str or datetime-like, optional
            Explicit start date (e.g. "2020-01-01"), used when
            ``time_range="buffered"``. Overrides ``scene_date``.
        t1 : str or datetime-like, optional
            Explicit end date (e.g. "2024-12-31").
            Defaults to present if only ``t0`` is provided.

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
            region = Raster(self.alt.dem_fn).get_bounds(latlon=True)

        # Resolve server-side time range to limit granules processed
        t0_str, t1_str, resolved_date = self._resolve_time_range(
            time_range=time_range,
            scene_date=scene_date,
            time_buffer_days=time_buffer_days,
            t0=t0,
            t1=t1,
        )
        self._print_time_filter_summary(
            time_range, t0_str, t1_str, resolved_date, t0, time_buffer_days
        )

        # Build the per-level SlideRule parameter sets (shared + custom).
        level_parms = self._build_level_parms(
            processing_levels, region, t0_str, t1_str, res, len, ats, maxi, cnt
        )

        # Record the request settings + cache locations so a report run can
        # write them to a figure-selections file for reproducibility (#121).
        self.atl06sr_request_parms = {
            "processing_levels": list(processing_levels),
            "res": res,
            "len": len,
            "ats": ats,
            "time_range": time_range,
            "t0": t0_str,
            "t1": t1_str,
        }
        self.atl06sr_parquet_paths = {}

        for key, parms in level_parms.items():
            print(f"\nICESat-2 ATL06 request processing for: {key}")
            fn = os.path.join(self.alt.directory, f"{filename}_{key}.parquet")
            self.atl06sr_parquet_paths[key] = fn
            print(parms)

            atl06sr = self._load_cached_atl06sr(fn, parms)
            if atl06sr is None:
                atl06sr = self._request_atl06sr_level(parms, fn, save_to_parquet)

            self._ingest_atl06sr(key, atl06sr, h_sigma_quantile)

    @staticmethod
    def _print_time_filter_summary(
        time_range, t0_str, t1_str, resolved_date, t0, time_buffer_days
    ):
        """Print the resolved server-side time filter, matching the request cascade."""
        if time_range == "all" and resolved_date is None and t0 is None:
            print(f"Time filter: {t0_str} to {t1_str} (all available)")
        elif resolved_date is not None:
            print(
                f"Time filter: {t0_str} to {t1_str} "
                f"(+/- {time_buffer_days} days from {resolved_date.date()})"
            )
        elif t0 is not None:
            print(f"Time filter: {t0_str} to {t1_str} (custom range)")
        else:
            print(f"Time filter: {t0_str} to {t1_str} (all available, fallback)")

    @staticmethod
    def _build_level_parms(
        processing_levels, region, t0_str, t1_str, res, len, ats, maxi, cnt
    ):
        """Build the merged SlideRule parameter dict for each requested level.

        Returns ``{level_key: parms}`` where ``parms`` is the shared request
        envelope merged with the level-specific confidence/surface-type
        settings. Only levels in ``processing_levels`` are included.

        See the SlideRule parameter discussion at
        https://github.com/SlideRuleEarth/sliderule/issues/448 — ``"srt": -1``
        tells the server to choose, per photon, the highest ATL03 confidence
        across all five surface-type entries. ``cnf`` options are
        ``atl03_{tep,not_considered,background,within_10m,low,medium,high}``.
        Ground uses a reduced ``cnt`` because ground photons are sparser.
        """
        # Shared parameters for all processing levels
        shared_parms = {
            "poly": region,
            "t0": t0_str,
            "t1": t1_str,
            "res": res,
            "len": len,
            "ats": ats,
            "fit": {"maxi": maxi},
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

        return {
            key: {**shared_parms, **parms}
            for key, parms in custom_parms.items()
            if key in processing_levels
        }

    def _load_cached_atl06sr(self, fn, parms):
        """Return cached ATL06-SR points if a parquet matching ``parms`` exists.

        Returns the cached :class:`~geopandas.GeoDataFrame` when ``fn`` exists
        and the SlideRule parameters it was built with match ``parms``; returns
        ``None`` (signalling a fresh request is needed) otherwise.
        """
        if not os.path.exists(fn):
            return None
        print(f"Existing file found, reading in: {fn}")
        atl06sr = gpd.read_parquet(fn)
        if self._params_match_cache(parms, atl06sr):
            return atl06sr
        return None

    @staticmethod
    def _params_match_cache(parms, atl06sr):
        """Whether a cached parquet was produced by the same SlideRule params.

        The request parameters are persisted in the parquet's
        ``sliderule_parameters`` column (see :meth:`_save_to_parquet`). The
        ``poly`` is compared as a string and the ``output`` key (a random temp
        path SlideRule injects during ``run()``) is stripped from both sides so
        an otherwise-identical request is recognized as a cache hit. Prints the
        reason and returns ``False`` when the cache cannot be reused.
        """
        if "sliderule_parameters" not in atl06sr.columns:
            print("No parameters column found, regenerating...")
            return False
        try:
            file_parms = json.loads(atl06sr["sliderule_parameters"].iloc[0])
        except Exception as e:
            print(f"Could not parse cached parameters: {e}. Regenerating...")
            return False

        parms_copy = parms.copy()
        parms_copy["poly"] = str(parms_copy["poly"])
        # Strip "output" (a random temp path injected by SlideRule during
        # run()) from both sides before comparison.
        parms_copy.pop("output", None)
        file_parms.pop("output", None)

        if str(parms_copy) == str(file_parms):
            return True
        print("Parameters don't match request. Regenerating...")
        return False

    def _request_atl06sr_level(self, parms, fn, save_to_parquet):
        """Fetch one processing level from SlideRule and sample WorldCover.

        Runs the ``atl03x`` request, samples ESA WorldCover before caching so
        the column is persisted in the parquet (and need not be re-sampled on
        later runs), and optionally writes the parquet to ``fn``.
        """
        atl06sr = sliderule_api.run("atl03x", parms)
        atl06sr = self._sample_worldcover_into_gdf(atl06sr)
        if save_to_parquet:
            self._save_to_parquet(fn, atl06sr, parms)
        return atl06sr

    def _ingest_atl06sr(self, key, atl06sr, h_sigma_quantile=1.0):
        """
        Normalize and quality-filter a raw ATL06-SR dataframe and store it.

        Shared by ``request_atl06sr_multi_processing`` (fresh request or cache
        hit) and ``load_atl06sr_from_parquet`` (replaying a prior run's points)
        so both paths produce identical ``atl06sr_processing_levels`` and
        ``atl06sr_processing_levels_filtered`` entries.

        Parameters
        ----------
        key : str
            Processing-level key (e.g. "all").
        atl06sr : geopandas.GeoDataFrame
            Raw ATL06-SR points (from SlideRule or a parquet cache).
        h_sigma_quantile : float, optional
            Quantile of ``h_sigma`` above which fits are discarded, default 1.0.
        """
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

    def load_atl06sr_from_parquet(self, parquet_paths, h_sigma_quantile=1.0):
        """
        Load ATL06-SR points directly from saved parquet caches.

        Replays the *exact* points a prior run used (issue #121), bypassing the
        SlideRule request entirely so a re-processed scene compares against an
        identical ICESat-2 sample. Runs the same normalization + quality filter
        as a fresh request via ``_ingest_atl06sr``.

        Parameters
        ----------
        parquet_paths : dict
            Mapping of processing-level key -> parquet path
            (e.g. ``{"all": ".../atl06sr_all.parquet"}``).
        h_sigma_quantile : float, optional
            Quantile of ``h_sigma`` above which fits are discarded, default 1.0.

        Returns
        -------
        bool
            True if at least one parquet was loaded, False otherwise.
        """
        self.atl06sr_parquet_paths = {}
        loaded_any = False
        for key, path in parquet_paths.items():
            if not path or not os.path.exists(path):
                logger.warning(
                    f"\nParquet for '{key}' not found at {path}; "
                    "cannot reuse those ICESat-2 points.\n"
                )
                continue
            print(f"Reusing ICESat-2 ATL06-SR points for '{key}' from: {path}")
            atl06sr = gpd.read_parquet(path)
            self.atl06sr_parquet_paths[key] = path
            self._restore_request_metadata_from_parquet(atl06sr)
            self._ingest_atl06sr(key, atl06sr, h_sigma_quantile)
            loaded_any = True
        return loaded_any

    def _restore_request_metadata_from_parquet(self, atl06sr):
        """
        Recover the SlideRule request time range from a cached parquet.

        The reuse path bypasses ``request_atl06sr_multi_processing`` (which sets
        ``self._t0`` / ``self._t1`` via ``_resolve_time_range``), so plot titles
        would otherwise lose their "<t0> to <t1>" date-range line. The request
        parameters are persisted in the parquet's ``sliderule_parameters``
        column, so read ``t0`` / ``t1`` back from there.
        """
        if hasattr(self, "_t0") and hasattr(self, "_t1"):
            return
        if "sliderule_parameters" not in atl06sr.columns or atl06sr.empty:
            return
        try:
            params = json.loads(atl06sr["sliderule_parameters"].iloc[0])
            t0, t1 = params.get("t0"), params.get("t1")
            if t0 and t1:
                self._t0 = pd.Timestamp(t0).to_pydatetime()
                self._t1 = pd.Timestamp(t1).to_pydatetime()
        except Exception as e:
            logger.warning(
                f"\nCould not restore time range from parquet metadata: {e}\n"
            )

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
        # SlideRule injects a random temp file path into parms["output"]
        # during run(); strip it so cached params are stable across runs.
        parms_copy.pop("output", None)
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

    @staticmethod
    def _worldcover_tile_url(lat, lon):
        """Return the AWS S3 URL for the ESA WorldCover tile covering (lat, lon)."""
        # Tiles are 3×3 degree; tile name = lower-left corner snapped to 3-degree grid
        tile_lat = int(np.floor(lat / 3.0) * 3)
        tile_lon = int(np.floor(lon / 3.0) * 3)
        ns = "N" if tile_lat >= 0 else "S"
        ew = "E" if tile_lon >= 0 else "W"
        name = f"{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}"
        return (
            f"https://esa-worldcover.s3.amazonaws.com/v200/2021/map/"
            f"ESA_WorldCover_10m_2021_v200_{name}_Map.tif"
        )

    def _sample_worldcover_into_gdf(self, atl06sr):
        """
        Sample ESA WorldCover 10m at each point in ``atl06sr`` and return
        a copy with an ``esa_worldcover.value`` column added.

        Reads Cloud Optimized GeoTIFFs directly from AWS S3. If the
        column already exists on the input, returns it unchanged.
        """
        import rasterio
        from rasterio.errors import RasterioIOError
        from rasterio.session import AWSSession

        wc_col = "esa_worldcover.value"
        if wc_col in atl06sr.columns:
            return atl06sr

        # The ESA WorldCover bucket is public, so read it anonymously
        # (aws_unsigned=True). Without this, rasterio creates a default
        # AWSSession that eagerly resolves credentials; on machines configured
        # with AWS SSO/login this raises botocore MissingDependencyException
        # ("requires botocore[crt]") and the whole report crashes — even though
        # no credentials are needed for a public bucket.
        unsigned_session = AWSSession(aws_unsigned=True)

        gdf_4326 = atl06sr.to_crs("EPSG:4326")
        lons = gdf_4326.geometry.x.values
        lats = gdf_4326.geometry.y.values

        tile_keys = set()
        for lat, lon in zip(lats, lons):
            tile_keys.add(self._worldcover_tile_url(lat, lon))

        print(
            f"  Sampling ESA WorldCover for {len(atl06sr)} points "
            f"({len(tile_keys)} tile(s))..."
        )

        values = np.full(len(atl06sr), np.nan)
        for url in tile_keys:
            try:
                with rasterio.Env(
                    unsigned_session,
                    GDAL_DISABLE_READDIR_ON_OPEN="YES",
                    CPL_VSIL_CURL_USE_HEAD="NO",
                ):
                    with rasterio.open(url) as src:
                        bounds = src.bounds
                        mask = (
                            (lons >= bounds.left)
                            & (lons < bounds.right)
                            & (lats >= bounds.bottom)
                            & (lats < bounds.top)
                        )
                        if not mask.any():
                            continue
                        coords = list(zip(lons[mask], lats[mask]))
                        sampled = np.array([v[0] for v in src.sample(coords)])
                        values[mask] = sampled
            except RasterioIOError as e:
                logger.warning(f"Could not read WorldCover tile {url}: {e}")

        atl06sr = atl06sr.copy()
        atl06sr[wc_col] = values

        n_valid = np.isfinite(values).sum()
        print(f"  WorldCover: {n_valid}/{len(atl06sr)} points sampled")
        return atl06sr

    def sample_esa_worldcover(self):
        """
        Sample ESA WorldCover 10m values into every filtered processing
        level that doesn't already have them.

        Normally not needed when data is requested via
        ``request_atl06sr_multi_processing``, which samples WorldCover
        automatically before caching. Use this when working with
        manually-loaded data.
        """
        for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
            self.atl06sr_processing_levels_filtered[key] = (
                self._sample_worldcover_into_gdf(atl06sr)
            )

    def filter_outliers(self, column="icesat_minus_dem", n_sigma=3):
        """
        Remove dh outliers beyond *n_sigma* × standard deviation from the mean.

        Parameters
        ----------
        column : str, optional
            Column to filter on, default ``"icesat_minus_dem"``.
        n_sigma : float, optional
            Number of standard deviations to allow, default 3.
        """
        for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
            if column not in atl06sr.columns:
                continue
            mask = self._std_outlier_mask(atl06sr[column], n_sigma)
            if mask is None:
                continue
            n_before = len(atl06sr)
            self.atl06sr_processing_levels_filtered[key] = atl06sr[mask]
            n_after = len(self.atl06sr_processing_levels_filtered[key])
            if n_before != n_after:
                print(
                    f"  Outlier filter ({n_sigma}σ): {key} {n_before} → {n_after} "
                    f"(removed {n_before - n_after})"
                )

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
            date = StereopairMetadataParser(self.alt.directory).get_pair_dict()["cdate"]
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
        df = atl06sr[["geometry", "h_mean"]].copy()
        df["lon"] = df["geometry"].x
        df["lat"] = df["geometry"].y
        df["height_above_datum"] = df["h_mean"]
        df = df[["lon", "lat", "height_above_datum"]]
        return self._write_csv_to_directory(df, f"{filename_prefix}_{key}.csv")

    def atl06sr_to_dem_dh(self, n_sigma=3):
        """
        Calculate height differences between ATL06-SR data and DEMs.

        Interpolates DEM heights at ATL06-SR point locations and calculates
        the difference between ICESat-2 heights and DEM heights. If an aligned
        DEM is available, also calculates differences against it. Outliers
        beyond ``n_sigma`` × NMAD from the median are removed by default.

        Parameters
        ----------
        n_sigma : float or None, optional
            Remove dh outliers beyond this many NMAD from the median.
            Default 3. Pass None to skip outlier filtering.

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
        # Sampling reprojects each track to the DEM CRS and stores the
        # reprojected frame back, so downstream track geometry (segment
        # endpoints) lives in the DEM/working CRS. The DEM is opened once and
        # interpolated per key.
        dem = self._open_dem(self.alt.dem_fn)
        for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
            sample, atl06sr = self._interp_dem_at_points(dem, atl06sr)
            atl06sr["dem_height"] = sample
            atl06sr["icesat_minus_dem"] = atl06sr["h_mean"] - atl06sr["dem_height"]
            self.atl06sr_processing_levels_filtered[key] = atl06sr

        if self.alt.aligned_dem_fn:
            aligned_dem = self._open_dem(self.alt.aligned_dem_fn)
            for key, atl06sr in self.atl06sr_processing_levels_filtered.items():
                sample, atl06sr = self._interp_dem_at_points(aligned_dem, atl06sr)
                atl06sr["aligned_dem_height"] = sample
                atl06sr["icesat_minus_aligned_dem"] = (
                    atl06sr["h_mean"] - atl06sr["aligned_dem_height"]
                )
                self.atl06sr_processing_levels_filtered[key] = atl06sr

        if n_sigma is not None:
            self.filter_outliers(n_sigma=n_sigma)

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

    def get_altimetry_selections(self, key="all"):
        """
        Report the ICESat-2 selections a run made, for reproducibility (#121).

        Bundles the request settings, the parquet cache locations (the exact
        points used), the auto-selected profile track, and the best/worst
        segment extents so they can be written to a figure-selections file and
        replayed on a later run.

        Parameters
        ----------
        key : str, optional
            Processing level key, default is "all".

        Returns
        -------
        dict
            ``{"request": {..}, "parquet_cache": {key: path},
            "profile_track": {"rgt", "cycle", "spot"},
            "segments": {"best": {..}, "worst": {..}}}``. Keys are omitted when
            the corresponding selection is unavailable.
        """
        selections = {}

        if getattr(self, "atl06sr_request_parms", None):
            selections["request"] = dict(self.atl06sr_request_parms)
        if getattr(self, "atl06sr_parquet_paths", None):
            selections["parquet_cache"] = dict(self.atl06sr_parquet_paths)

        resolved = self._resolve_best_track(key)
        if resolved is None:
            return selections
        track, rgt, cycle, spot = resolved[0], resolved[1], resolved[2], resolved[3]
        selections["profile_track"] = {
            "rgt": int(rgt),
            "cycle": int(cycle),
            "spot": int(spot),
        }

        seg_info = self._find_best_worst_segments(track)
        if seg_info is not None:
            selections["segments"] = {
                "best": self._segment_record(track, seg_info, "best"),
                "worst": self._segment_record(track, seg_info, "worst"),
            }
        return selections

    @staticmethod
    def _segment_record(track, seg_info, which):
        """
        Build a serializable record for a best/worst segment, used by
        ``get_altimetry_selections``.

        ``start_xatc`` / ``end_xatc`` are the **absolute** along-track extents
        (meters) and are what reuse actually replays — absolute ``x_atc`` is
        stable even when outlier filtering drops a different first point, whereas
        a track-start-relative offset would shift. ``start_km`` / ``end_km``
        (km from the track start) are kept for human readability. ``endpoints_xy``
        stores the segment's first/last point coordinates **in the track
        geometry's CRS** (the DEM/working CRS after dh computation, typically UTM
        — not lon/lat), for human inspection only.
        """
        record = {
            "start_xatc": float(seg_info[f"seg_{which}_start_xatc"]),
            "end_xatc": float(seg_info[f"seg_{which}_end_xatc"]),
            "start_km": float(seg_info[f"seg_{which}_start_km"]),
            "end_km": float(seg_info[f"seg_{which}_end_km"]),
        }
        seg_pts = track[seg_info[f"seg_{which}_mask"]]
        if not seg_pts.empty and seg_pts.geometry.notna().any():
            geom = seg_pts.geometry
            record["endpoints_xy"] = [
                [float(geom.iloc[0].x), float(geom.iloc[0].y)],
                [float(geom.iloc[-1].x), float(geom.iloc[-1].y)],
            ]
        return record

    def _resolve_best_track(self, key="all", rgt=None, cycle=None, spot=None):
        """
        Resolve track selection and return the filtered, sorted track DataFrame.

        Returns
        -------
        tuple of (track, rgt, cycle, spot, track_count, track_date, dist, dh_vals)
            or None if no valid track found.
        """
        if not all([rgt, cycle, spot]):
            best = self._select_best_track(key)
            if best is None:
                logger.warning("\nNo valid tracks found for profile plot. Skipping.\n")
                return None
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
                f"\nNo data for RGT={rgt}, Cycle={cycle}, Spot={spot}. Skipping.\n"
            )
            return None

        track = track.sort_values("x_atc")
        dist = (track["x_atc"] - track["x_atc"].min()) / 1000.0

        if track_date is None:
            track_date = track.index[0].strftime("%Y-%m-%d")
        if track_count is None:
            track_count = len(track)

        dh_vals = track["icesat_minus_dem"].dropna()

        return (track, rgt, cycle, spot, track_count, track_date, dist, dh_vals)

    def _find_best_worst_segments(
        self, track, dh_col="icesat_minus_dem", segment_override=None
    ):
        """
        Identify 1 km segments with better and worse agreement along a track.

        Agreement is scored as ``3·|median(dh)| + NMAD(dh)``, weighting the
        median bias three times more than the dispersion so that a segment
        with a large bias cannot be selected as "better agreement" just
        because its NMAD is small.

        Parameters
        ----------
        track : geopandas.GeoDataFrame
            The resolved track (sorted by ``x_atc``).
        dh_col : str, optional
            Column of height differences to score, default "icesat_minus_dem".
        segment_override : dict or None, optional
            When provided, pins the segment extents instead of scoring them
            (issue #121). Expects ``{"best": {...}, "worst": {...}}`` where each
            entry carries **absolute** along-track extents ``start_xatc`` /
            ``end_xatc`` (meters). ``start_km`` / ``end_km`` (km from the track
            start) are accepted as a legacy fallback, but absolute ``x_atc`` is
            preferred because it is stable even when outlier filtering drops a
            different first point and shifts the track start.

        Returns
        -------
        dict or None
            Dictionary with keys: seg_best_mask, seg_worst_mask,
            seg_{best,worst}_{start,end}_km (relative to this track's start, for
            plotting) and seg_{best,worst}_{start,end}_xatc (absolute, for
            stable reuse). None if segments cannot be identified.
        """
        x_atc = track["x_atc"].values
        track_length_m = x_atc[-1] - x_atc[0] if len(x_atc) > 1 else 0
        diffs = np.diff(x_atc)
        median_spacing = np.median(diffs) if len(diffs) > 0 else 0
        if not (median_spacing > 0 and track_length_m >= 1000):
            return None

        # Pinned segments: rebuild the masks/extents from absolute along-track
        # (x_atc) positions so a re-run highlights the same ground segments even
        # if outlier filtering removed a different first point (which would shift
        # any track-start-relative km offset).
        if segment_override is not None:
            try:
                x_atc_lo = x_atc[0]

                def _resolve_extent(seg):
                    if seg.get("start_xatc") is not None and (
                        seg.get("end_xatc") is not None
                    ):
                        return float(seg["start_xatc"]), float(seg["end_xatc"])
                    # Legacy fallback: km relative to track start.
                    return (
                        x_atc_lo + float(seg["start_km"]) * 1000.0,
                        x_atc_lo + float(seg["end_km"]) * 1000.0,
                    )

                bs, be = _resolve_extent(segment_override["best"])
                ws, we = _resolve_extent(segment_override["worst"])
                return self._segment_dict(track, bs, be, ws, we)
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(
                    f"\nCould not apply pinned segments ({e}); "
                    "falling back to automatic segment selection.\n"
                )

        half_win = 500  # meters
        median_weight = 3.0  # weight |median(dh)| more heavily than NMAD
        scores = []
        for xc in x_atc:
            mask_i = (track["x_atc"] >= xc - half_win) & (
                track["x_atc"] <= xc + half_win
            )
            seg_i = track.loc[mask_i]
            n_total = len(seg_i)
            if n_total < 3:
                scores.append(np.nan)
                continue
            dem_valid = seg_i["dem_height"].notna().sum()
            if dem_valid / n_total < 0.75:
                scores.append(np.nan)
                continue
            seg_dh = seg_i[dh_col].dropna().values
            if len(seg_dh) < 3:
                scores.append(np.nan)
                continue
            scores.append(median_weight * abs(np.nanmedian(seg_dh)) + _nmad(seg_dh))

        scores = np.array(scores)
        if (~np.isnan(scores)).sum() < 2:
            return None

        idx_best = np.nanargmin(scores)
        idx_worst = np.nanargmax(scores)
        x_atc_lo, x_atc_hi = x_atc[0], x_atc[-1]

        seg_best_start = max(x_atc[idx_best] - half_win, x_atc_lo)
        seg_best_end = min(x_atc[idx_best] + half_win, x_atc_hi)
        seg_worst_start = max(x_atc[idx_worst] - half_win, x_atc_lo)
        seg_worst_end = min(x_atc[idx_worst] + half_win, x_atc_hi)

        return self._segment_dict(
            track, seg_best_start, seg_best_end, seg_worst_start, seg_worst_end
        )

    @staticmethod
    def _segment_dict(track, bs, be, ws, we):
        """
        Build the best/worst segment dict from absolute along-track extents.

        Parameters
        ----------
        track : geopandas.GeoDataFrame
            Resolved track (sorted by ``x_atc``).
        bs, be, ws, we : float
            Absolute ``x_atc`` (meters) start/end of the best and worst segments.

        Returns
        -------
        dict
            Masks, km extents (relative to this track's start, for plotting),
            and absolute ``x_atc`` extents (for stable reuse).
        """
        x_atc_lo = track["x_atc"].values[0]
        return {
            "seg_best_mask": (track["x_atc"] >= bs) & (track["x_atc"] <= be),
            "seg_worst_mask": (track["x_atc"] >= ws) & (track["x_atc"] <= we),
            "seg_best_start_km": (bs - x_atc_lo) / 1000.0,
            "seg_best_end_km": (be - x_atc_lo) / 1000.0,
            "seg_worst_start_km": (ws - x_atc_lo) / 1000.0,
            "seg_worst_end_km": (we - x_atc_lo) / 1000.0,
            "seg_best_start_xatc": float(bs),
            "seg_best_end_xatc": float(be),
            "seg_worst_start_xatc": float(ws),
            "seg_worst_end_xatc": float(we),
        }
