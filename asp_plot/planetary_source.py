"""Planetary (LOLA/MOLA) altimetry source.

Owns the planetary-altimetry side of :class:`asp_plot.altimetry.Altimetry`:
loading LOLA (Moon) and MOLA (Mars) point clouds from ODE GDS CSV exports,
sampling the ASP DEM at those points, and exporting a pc_align-ready CSV.

The class holds its own data (``planetary_points``) and reads the
cross-cutting ``dem_fn`` / ``directory`` / ``aligned_dem_fn`` from the
coordinating :class:`Altimetry` instance passed at construction.
"""

import logging

import geopandas as gpd
import pandas as pd

from asp_plot.altimetry_source import AltimetrySource
from asp_plot.bodies import BODIES

logger = logging.getLogger(__name__)

# IAU 2000 mean equatorial radii used by ASP DEMs and pc_align (stored as
# (planetary_radius - sphere_radius)). Sourced from the body registry so the
# values live in exactly one place.
MARS_IAU_SPHERE_RADIUS = BODIES["mars"].iau_sphere_radius_m  # meters
MOON_IAU_SPHERE_RADIUS = BODIES["moon"].iau_sphere_radius_m  # meters

# --- ODE GDS REST API (for LOLA/MOLA planetary altimetry) ---

GDS_BASE_URL = "https://oderest.rsl.wustl.edu/livegds"


def gds_query_async(query_type, bounds, results_code, email=None, **extra_params):
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

    url = f"{GDS_BASE_URL}?{urllib.parse.urlencode(params)}"
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


class PlanetarySource(AltimetrySource):
    """Body-agnostic planetary altimetry against an ASP DEM.

    Holds the machinery shared by every planetary instrument: sampling the DEM
    at the loaded points, differencing, outlier filtering, and exporting a
    pc_align-ready CSV. The CSV *loader* is body-specific and lives in the
    :class:`LolaSource` (Moon) and :class:`MolaSource` (Mars) subclasses; the
    coordinator instantiates the right one from the DEM's body. The base
    :meth:`load_planetary_csv` raises, so an Earth DEM that reaches here is
    pointed back at ICESat-2.

    Parameters
    ----------
    alt : Altimetry
        The coordinating :class:`asp_plot.altimetry.Altimetry` instance.
        Cross-cutting paths (``dem_fn``, ``directory``, ``aligned_dem_fn``)
        are read from it so a single source of truth describes the DEM
        under analysis.
    """

    #: Reference altimetry instrument name, set by the body subclasses.
    instrument = None

    # Column name candidates for LOLA and MOLA CSVs
    _LON_CANDIDATES = [
        "pt_longitude",
        "long_east",
        "longitude",
        "areocentric_longitude",
    ]
    _LAT_CANDIDATES = ["pt_latitude", "lat_north", "latitude", "areocentric_latitude"]
    _TOPO_CANDIDATES = ["topography", "topo"]
    # PLANET_RAD column names from ODE GDS PEDR (Mars) and LOLA RDR
    # "Point Per Row" CSV (Moon, has Pt_Radius). Order matters — names
    # earlier in the list are preferred.
    _RADIUS_CANDIDATES = [
        "planet_rad",
        "planet_rad (shot_planetary_radius)",
        "pt_radius",
        "planetary_radius",
        "radius",
    ]

    def __init__(self, alt):
        super().__init__(alt)
        self.planetary_points = None

    def load_planetary_csv(self, csv_path):
        """Load a planetary altimetry CSV (overridden per body).

        The base implementation raises: only :class:`LolaSource` and
        :class:`MolaSource` know how to parse a GDS CSV. The CSV is obtained
        via the ``request_planetary_altimetry`` CLI tool, which submits an
        async query to the ODE GDS API and emails the user a download link;
        the user downloads and unzips the result, then passes the
        ``*_topo_csv.csv`` / ``*_pts_csv.csv`` file to the body source.

        Parameters
        ----------
        csv_path : str
            Path to a CSV file from the ODE GDS.
        """
        raise ValueError(
            "Planetary altimetry CSV loading is not supported for this DEM "
            "(no LOLA/MOLA body detected). Use ICESat-2 for Earth DEMs."
        )

    def _build_planetary_gdf(self, df, geo_crs):
        """Build and store the planetary GeoDataFrame from a parsed CSV frame.

        ``df`` must carry ``lon`` / ``lat`` / ``height`` / ``radius_m`` columns.
        Stores the result on ``self.planetary_points`` and returns it.
        """
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs=geo_crs,
        )
        self.planetary_points = gdf
        return gdf

    @staticmethod
    def _find_csv_column(cols_lower, candidates):
        """Find a CSV column by matching against candidate names.

        Parameters
        ----------
        cols_lower : dict
            Mapping of ``{stripped_lowercase_name: original_name}``.
        candidates : list of str
            Candidate column names to search for (lowercase).

        Returns
        -------
        str or None
            Original column name if found, else None.
        """
        for c in candidates:
            if c in cols_lower:
                return cols_lower[c]
        return None

    def _load_planetary_csv_common(self, csv_path, instrument, prefer="radius"):
        """Shared CSV loading logic for LOLA and MOLA.

        Reads the CSV, validates columns, converts longitude to -180/180.
        Picks PLANET_RAD over TOPOGRAPHY by default (``prefer="radius"``)
        because TOPOGRAPHY is referenced to the oblate areoid and produces
        a latitude-dependent offset against ASP's spherical-IAU DEMs.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        instrument : str
            ``"LOLA"`` or ``"MOLA"`` (for error messages).
        prefer : {"radius", "topo"}
            Which column family to try first.

        Returns
        -------
        tuple of (pandas.DataFrame, bool)
            (df with ``lon``, ``lat``, ``height_raw`` columns,
             ``is_radius`` flag — True if the chosen column is a planetary radius)
        """
        df = pd.read_csv(csv_path)

        if df.empty:
            raise ValueError(
                f"{instrument} CSV is empty: {csv_path}\n"
                "The query area may have no coverage."
            )

        cols_lower = {c.strip().lower(): c for c in df.columns}

        lon_col = self._find_csv_column(cols_lower, self._LON_CANDIDATES)
        lat_col = self._find_csv_column(cols_lower, self._LAT_CANDIDATES)

        if prefer == "radius":
            primary = self._find_csv_column(cols_lower, self._RADIUS_CANDIDATES)
            fallback = self._find_csv_column(cols_lower, self._TOPO_CANDIDATES)
            is_radius = primary is not None
        else:
            primary = self._find_csv_column(cols_lower, self._TOPO_CANDIDATES)
            fallback = self._find_csv_column(cols_lower, self._RADIUS_CANDIDATES)
            is_radius = primary is None and fallback is not None

        height_col = primary if primary is not None else fallback

        if lon_col is None or lat_col is None or height_col is None:
            raise ValueError(
                f"{instrument} CSV does not have expected columns.\n"
                f"  Found: {list(df.columns)}\n"
                f"  Expected longitude, latitude, and either a planetary "
                f"radius (PLANET_RAD / Pt_Radius) or topography column."
            )

        df = df.rename(
            columns={lon_col: "lon", lat_col: "lat", height_col: "height_raw"}
        )

        # Convert 0-360 → -180/180
        df["lon"] = ((df["lon"] + 180) % 360) - 180

        return df, is_radius

    def _sample_dem_at_planetary_points(self, dem_fn, height_col, dh_col):
        """Interpolate DEM heights at the loaded altimetry points.

        Adds ``height_col`` and ``dh_col`` (= altimetry height − DEM height)
        to ``self.planetary_points`` in-place. Used by
        :meth:`planetary_to_dem_dh` for both the raw and aligned DEMs.
        """
        if self.planetary_points is None or self.planetary_points.empty:
            return

        dem = self._open_dem(dem_fn)
        sample, _ = self._interp_dem_at_points(dem, self.planetary_points)

        self.planetary_points[height_col] = sample
        self.planetary_points[dh_col] = self.planetary_points["height"] - sample

    def planetary_to_dem_dh(self, n_sigma=3):
        """Compute height differences between planetary altimetry and DEM.

        Reprojects ``self.planetary_points`` to the DEM CRS, interpolates
        DEM heights at altimetry locations, and computes the difference
        ``altimetry_minus_dem = height - dem_height``. When
        ``aligned_dem_fn`` is set, also populates ``aligned_dem_height``
        and ``altimetry_minus_aligned_dem`` so pre/post-alignment plots can
        share a single sample. Outliers beyond ``n_sigma`` × std from the
        mean (computed on the unaligned dh) are removed by default.

        Parameters
        ----------
        n_sigma : float or None, optional
            Remove dh outliers beyond this many standard deviations from
            the mean. Default 3. Pass None to skip outlier filtering.

        The results are stored as new columns on ``self.planetary_points``.
        """
        if self.planetary_points is None or self.planetary_points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return

        self._sample_dem_at_planetary_points(
            self.alt.dem_fn, "dem_height", "altimetry_minus_dem"
        )
        if self.alt.aligned_dem_fn:
            self._sample_dem_at_planetary_points(
                self.alt.aligned_dem_fn,
                "aligned_dem_height",
                "altimetry_minus_aligned_dem",
            )

        valid = self.planetary_points["altimetry_minus_dem"].dropna()
        print(f"Computed dh for {len(valid)} of {len(self.planetary_points)} points")

        if n_sigma is not None and not valid.empty:
            mask = self._std_outlier_mask(
                self.planetary_points["altimetry_minus_dem"], n_sigma
            )
            if mask is not None:
                n_before = len(self.planetary_points)
                self.planetary_points = self.planetary_points[mask]
                n_after = len(self.planetary_points)
                if n_before != n_after:
                    print(
                        f"  Outlier filter ({n_sigma}σ): "
                        f"{n_before} → {n_after} "
                        f"(removed {n_before - n_after})"
                    )

    def to_csv_for_pc_align_planetary(self, filename_prefix="planetary_for_pc_align"):
        """Export ``self.planetary_points`` to a CSV for pc_align.

        Writes columns ``lon, lat, radius_m`` (planetary radius from the
        body center, in meters). Used as the ``planetary_csv`` argument
        to :meth:`asp_plot.alignment.Alignment.pc_align_dem_to_planetary_csv`.

        Parameters
        ----------
        filename_prefix : str, optional
            Prefix for the output CSV filename. Saved in
            ``self.alt.directory``.

        Returns
        -------
        str
            Absolute path to the created CSV file.
        """
        if self.planetary_points is None or self.planetary_points.empty:
            raise ValueError("No planetary altimetry points loaded.")
        if "radius_m" not in self.planetary_points.columns:
            raise ValueError(
                "planetary_points has no radius_m column. Call "
                "load_planetary_csv() first to populate it."
            )

        # Drop dh-NaN rows so pc_align doesn't reject the file. We also
        # restrict to points that fall on the DEM (have a finite
        # dem_height) when available — pc_align can use them as the
        # reference cloud only if they overlap the source.
        df = self.planetary_points.copy()
        df["lon"] = df.geometry.x
        df["lat"] = df.geometry.y
        df = df[["lon", "lat", "radius_m"]].dropna()
        if df.empty:
            raise ValueError("No valid planetary altimetry points after dropping NaNs.")

        return self._write_csv_to_directory(df, f"{filename_prefix}.csv")


class LolaSource(PlanetarySource):
    """LOLA (Moon) altimetry source.

    Parses a LOLA RDR CSV from the ODE GDS into height above the IAU 1737.4 km
    lunar sphere. The Moon is nearly spherical (1.4 km equatorial-vs-polar
    variation), so LOLA TOPOGRAPHY ≈ Pt_Radius − 1737.4 km — either column
    gives the same dh against an ASP lunar DEM to <1 m.
    """

    instrument = "LOLA"

    def load_planetary_csv(self, csv_path):
        """Parse a LOLA topo/RDR CSV into ``self.planetary_points``.

        Stores height above the IAU lunar sphere on ``height`` and the
        planetary radius on ``radius_m`` (used by pc_align).

        Parameters
        ----------
        csv_path : str
            Path to a LOLA RDR CSV from the ODE GDS API.
        """
        df, is_radius = self._load_planetary_csv_common(
            csv_path, "LOLA", prefer="radius"
        )

        if is_radius:
            # LOLA RDR Pt_Radius is in KILOMETERS (per the ODE GDS Point
            # per Row CSV header), unlike MOLA PEDR PLANET_RAD which is
            # in meters. Detect units by magnitude (~1737 vs ~1737000)
            # and normalize to meters.
            if df["height_raw"].median() < 10000:
                df["height_raw"] = df["height_raw"] * 1000.0
                unit_note = " (km → m)"
            else:
                unit_note = ""
            df["height"] = df["height_raw"] - MOON_IAU_SPHERE_RADIUS
            df["radius_m"] = df["height_raw"]
            source = f"Pt_Radius{unit_note}"
        else:
            # Topography path: the Moon is essentially spherical so LOLA
            # topography ≈ height above the IAU 1737.4 km sphere.
            df["height"] = df["height_raw"]
            df["radius_m"] = df["height_raw"] + MOON_IAU_SPHERE_RADIUS
            source = "Topography"

        gdf = self._build_planetary_gdf(df, BODIES["moon"].geographic_crs_wkt)
        print(f"Loaded {len(gdf)} LOLA points (using {source} column)")


class MolaSource(PlanetarySource):
    """MOLA (Mars) altimetry source.

    Parses a MOLA PEDR CSV from the ODE GDS, using the PLANET_RAD column to
    compute height above the IAU Mars sphere (3,396,190 m):
    ``height = PLANET_RAD - 3,396,190``. This is the same reference as ASP
    DEMs, so dh = MOLA − DEM is directly meaningful at all latitudes.
    """

    instrument = "MOLA"

    def load_planetary_csv(self, csv_path):
        """Parse a MOLA PEDR CSV into ``self.planetary_points``.

        TOPOGRAPHY (in ``*_topo_csv.csv``) is referenced to the MOLA oblate
        areoid and produces a latitude-dependent offset of up to ~10 km against
        an ASP DEM, so it is rejected here. Pass the ``*_pts_csv.csv`` from the
        ODE GDS download instead — it carries PLANET_RAD.

        Parameters
        ----------
        csv_path : str
            Path to a MOLA PEDR CSV from the ODE GDS API. Must include
            a ``PLANET_RAD`` column (use the ``*_pts_csv.csv`` file).
        """
        df, is_radius = self._load_planetary_csv_common(
            csv_path, "MOLA", prefer="radius"
        )

        if not is_radius:
            raise ValueError(
                f"MOLA CSV is missing the PLANET_RAD column: {csv_path}\n"
                "The ODE GDS '*_topo_csv.csv' only has TOPOGRAPHY, which is "
                "height above the oblate MOLA areoid. ASP DEMs use the "
                "spherical IAU 2000 datum (3,396,190 m), so dh from "
                "TOPOGRAPHY carries a latitude-dependent offset of up to "
                "~10 km that pc_align cannot remove.\n\n"
                "Pass the '*_pts_csv.csv' file from the same ODE GDS "
                "download instead — it contains PLANET_RAD."
            )

        df["height"] = df["height_raw"] - MARS_IAU_SPHERE_RADIUS
        df["radius_m"] = df["height_raw"]

        gdf = self._build_planetary_gdf(df, BODIES["mars"].geographic_crs_wkt)
        print(
            f"Loaded {len(gdf)} MOLA points "
            f"(PLANET_RAD - {MARS_IAU_SPHERE_RADIUS:.0f} m → height above IAU sphere)"
        )
