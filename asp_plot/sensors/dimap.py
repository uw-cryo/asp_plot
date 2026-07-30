"""Metadata reader for Airbus-family DIMAP v2 camera files.

Covers the sensors that share the DIMAP v2 primary-product layout: Pléiades
1A/1B, Pléiades Neo, SPOT 6/7, and PeruSat-1. ASP parses these with a single
reader too (PleiadesXML.cc; PeruSatXML.cc duplicates the same layout), which
is the reference implementation this module mirrors.
"""

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon

from asp_plot.sensors.base import SensorMetadata, _common_base, fill_scene_defaults

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# DIMAP METADATA_PROFILE values this reader supports, matching ASP's own
# coverage of the layout: Pléiades 1A/1B (PHR), Pléiades Neo (PNEO), and
# SPOT 6/7 via PleiadesXML.cc, PeruSat-1 via PeruSatXML.cc. Any other DIMAP
# product profile gets an explicit "unsupported profile" warning instead of
# a wrong parse or a confusing fall-through.
SUPPORTED_DIMAP_PROFILES = (
    "PHR_SENSOR",
    "PNEO_SENSOR",
    "S6_SENSOR",
    "S7_SENSOR",
    "PER1_SENSOR",
)

# Profiles implemented from the ASP reader spec (PleiadesXML.cc /
# PeruSatXML.cc) but not yet validated against a real delivery (#168):
# parsing one emits a one-time informational warning asking for reports.
SPEC_ONLY_DIMAP_PROFILES = ("S6_SENSOR", "S7_SENSOR", "PER1_SENSOR")

# Unsupported-profile warnings already emitted, keyed by absolute path:
# detection can inspect the same file several times (shallow and recursive
# passes, then reader construction), and the hint is only useful once.
_warned_unsupported_profiles = set()

# Spec-only profiles already warned about, keyed by profile: the caveat
# applies to the profile as a whole, not to individual files.
_warned_spec_only_profiles = set()


class PleiadesMetadata(SensorMetadata):
    """Metadata reader for Airbus-family DIMAP v2 camera files.

    Parses DIMAP v2 primary-product metadata (``DIM_*.XML``, root tag
    ``Dimap_Document``) as delivered with Pléiades 1A/1B, Pléiades Neo,
    SPOT 6/7, and PeruSat-1 SEN(sor) products. Each scene is delivered as a
    single DIM XML, so unlike WorldView there is no tile mosaicking step. The
    sidecar ``RPC_*.XML`` files share the DIMAP root but carry no ephemeris,
    attitude, or acquisition-angle information, so discovery keeps only files
    whose ``METADATA_SUBPROFILE`` is ``PRODUCT`` and whose
    ``METADATA_PROFILE`` is one of ``SUPPORTED_DIMAP_PROFILES`` (products
    from other DIMAP profiles are skipped with an explanatory warning).
    Profiles in ``SPEC_ONLY_DIMAP_PROFILES`` are implemented from ASP's
    reader spec but not yet validated against real deliveries (#168).

    Notes
    -----
    - Airbus quaternions are scalar-first (``Q0`` is the scalar part). They are
      reordered to the scalar-last ``q1..q4`` layout shared with WorldView, as
      consumed by the roll/pitch/yaw computation in
      :meth:`asp_plot.stereo_geometry.StereoGeometryPlotter._compute_roll_pitch_yaw`.
    - Attitude comes in two shapes: tabulated quaternion samples
      (``Quaternion_List``; Pléiades Neo, SPOT 6/7, PeruSat-1) or one degree-3
      polynomial per quaternion component (``Polynomial_Quaternions``;
      Pléiades 1A/1B), which is evaluated at the ephemeris timestamps so both
      shapes yield the same tabulated ``att_df`` downstream.
    - DIMAP reports no ephemeris/attitude covariance and no scan direction or
      TDI level: the ``cov_*`` columns are filled with NaN and ``scandir`` /
      ``tdi`` are None. Consumers treat those as "not provided".
    - The mean view/sun angles and GSD are averaged over the
      ``Located_Geometric_Values`` blocks — nine for Pléiades/SPOT (corners,
      edge midpoints, center), a single center block for PeruSat-1.
      ``meansatel`` is derived as 90° minus the mean target incidence angle,
      matching the WorldView ``MEANSATEL`` convention.
    """

    name = "Pleiades"

    def __init__(self, directory=None, image_list=None):
        """
        Initialize the Pléiades metadata reader.

        The reader can be built either from a ``directory`` (its DIMAP product
        XMLs are discovered) or from an explicit ``image_list`` of XML files.
        At least one of the two must be given. Non-product DIMAP XMLs (RPC,
        LUT, volume indexes) are filtered out in both cases.

        Parameters
        ----------
        directory : str, optional
            Path to directory containing DIMAP camera metadata files.
        image_list : list of str, optional
            Explicit list of XML files to use instead of discovering them
            from ``directory``.

        Raises
        ------
        ValueError
            If neither ``directory`` nor ``image_list`` is given, or if no
            DIMAP product XML files are found.
        """
        if directory is None and image_list is None:
            raise ValueError("Provide either a directory or an image_list.")

        if image_list is not None:
            self.image_list = self._filter_camera_xmls(image_list)
            self.directory = (
                os.path.expanduser(directory)
                if directory
                else _common_base(self.image_list)
            )
        else:
            super().__init__(directory)
            self.image_list = self._discover_xmls(self.directory)

        if not self.image_list:
            raise ValueError(
                "\n\nMissing DIMAP (DIM_*.XML) camera metadata files. "
                "Cannot extract metadata without these.\n\n"
            )

    @staticmethod
    def _dimap_profiles(xml_fn):
        """Return ``(profile, subprofile)`` of a DIMAP XML, or None.

        Reads ``Metadata_Identification/METADATA_PROFILE`` and
        ``METADATA_SUBPROFILE`` with ``iterparse``, stopping as soon as both
        are seen (they sit near the top of the file), so inspection stays
        cheap even though DIM files run to several MB. Returns None for
        non-DIMAP or unparseable XML.
        """
        try:
            profile = subprofile = None
            root_seen = False
            for event, el in ET.iterparse(xml_fn, events=("start", "end")):
                if not root_seen:
                    if el.tag != "Dimap_Document":
                        return None
                    root_seen = True
                elif event == "end":
                    if el.tag == "METADATA_PROFILE":
                        profile = (el.text or "").strip()
                    elif el.tag == "METADATA_SUBPROFILE":
                        subprofile = (el.text or "").strip()
                    if profile is not None and subprofile is not None:
                        return (profile, subprofile)
            return (profile, subprofile) if root_seen else None
        except (ET.ParseError, OSError):
            return None

    @classmethod
    def _is_camera_file(cls, path):
        """True if ``path`` is a DIMAP *product* file of a supported profile.

        A camera scene's metadata is the DIMAP file whose
        ``METADATA_SUBPROFILE`` is ``PRODUCT`` (the ``DIM_*.XML``); the RPC
        sidecars (subprofile ``RPC``) and any non-DIMAP XML are rejected
        silently. A product whose ``METADATA_PROFILE`` is *not* in
        ``SUPPORTED_DIMAP_PROFILES`` is also rejected, but with a one-time
        warning naming the profile, so the user learns why the file was
        skipped instead of hitting a generic "no supported sensor" error.
        """
        profiles = cls._dimap_profiles(path)
        if profiles is None:
            return False
        profile, subprofile = profiles
        if subprofile != "PRODUCT":
            return False
        if profile not in SUPPORTED_DIMAP_PROFILES:
            key = os.path.abspath(path)
            if key not in _warned_unsupported_profiles:
                _warned_unsupported_profiles.add(key)
                logger.warning(
                    "DIMAP product with unsupported METADATA_PROFILE '%s' "
                    "skipped: %s (supported: %s; broader DIMAP-family support "
                    "is tracked in uw-cryo/asp_plot#168)",
                    profile,
                    path,
                    ", ".join(SUPPORTED_DIMAP_PROFILES),
                )
            return False
        return True

    def get_scene_dicts(self):
        """Return one sensor-agnostic scene dict per DIMAP product XML."""
        return [self.get_scene_dict(xml) for xml in self.image_list]

    def get_scene_dict(self, xml, geteph=True):
        """
        Get a dictionary of metadata for one DIMAP scene.

        Parameters
        ----------
        xml : str
            Path to the ``DIM_*.XML`` product metadata file.
        geteph : bool, optional
            Whether to extract ephemeris/attitude data, default is True.

        Returns
        -------
        dict
            Sensor-agnostic scene dict (see package docstring).
        """
        root = ET.parse(xml).getroot()

        profile = root.findtext(".//Metadata_Identification/METADATA_PROFILE")
        if (
            profile in SPEC_ONLY_DIMAP_PROFILES
            and profile not in _warned_spec_only_profiles
        ):
            _warned_spec_only_profiles.add(profile)
            logger.warning(
                "DIMAP profile '%s' support is implemented from the ASP "
                "reader spec but not yet validated against real data — "
                "please report issues at "
                "https://github.com/uw-cryo/asp_plot/issues/168",
                profile,
            )

        lgvs = root.findall(".//Geometric_Data/Use_Area/Located_Geometric_Values")

        def lgv_mean(path):
            vals = [
                float(lgv.findtext(path))
                for lgv in lgvs
                if lgv.findtext(path) is not None
            ]
            return np.round(np.mean(vals), 2) if vals else np.nan

        start = root.findtext(".//Refined_Model/Time/Time_Range/START")
        date = datetime.fromisoformat(start.replace("Z", "+00:00")).replace(tzinfo=None)

        mission = root.findtext(".//Strip_Source/MISSION") or (
            profile or "DIMAP"
        ).replace("_SENSOR", "")
        mission_index = root.findtext(".//Strip_Source/MISSION_INDEX") or ""

        verts = root.findall(".//Dataset_Extent/Vertex")
        geom = Polygon(
            [(float(v.findtext("LON")), float(v.findtext("LAT"))) for v in verts]
        )

        cloudcover = root.findtext(".//Dataset_Content/CLOUD_COVERAGE")

        # Mean product GSD: average of the along- and across-track GSDs over
        # the located-values grid (Pléiades products are near-square pixels).
        meanproductgsd = np.round(
            np.nanmean(
                [
                    lgv_mean("Ground_Sample_Distance/GSD_ACROSS_TRACK"),
                    lgv_mean("Ground_Sample_Distance/GSD_ALONG_TRACK"),
                ]
            ),
            2,
        )

        d = {
            "xml_fn": xml,
            "catid": root.findtext(".//Dataset_Identification/DATASET_NAME")
            or os.path.splitext(os.path.basename(xml))[0],
            "sensor": f"{mission}{mission_index}",
            "date": date,
            "scandir": None,
            "tdi": None,
            "geom": geom,
            "meansataz": lgv_mean("Acquisition_Angles/AZIMUTH_ANGLE"),
            "meansatel": np.round(
                90.0 - lgv_mean("Acquisition_Angles/INCIDENCE_ANGLE"), 2
            ),
            "meanoffnadirviewangle": lgv_mean("Acquisition_Angles/VIEWING_ANGLE"),
            "meanintrackviewangle": lgv_mean(
                "Acquisition_Angles/VIEWING_ANGLE_ALONG_TRACK"
            ),
            "meancrosstrackviewangle": lgv_mean(
                "Acquisition_Angles/VIEWING_ANGLE_ACROSS_TRACK"
            ),
            "meanproductgsd": meanproductgsd,
            "meansunaz": lgv_mean("Solar_Incidences/SUN_AZIMUTH"),
            "meansunel": lgv_mean("Solar_Incidences/SUN_ELEVATION"),
            "cloudcover": float(cloudcover) if cloudcover is not None else np.nan,
        }

        if geteph:
            d["eph_gdf"] = self.getEphem_gdf(root)
            d["att_df"] = self.getAtt_df(root)
            d["fp_gdf"] = gpd.GeoDataFrame(
                {"idx": [0], "geometry": [geom]},
                geometry="geometry",
                crs="EPSG:4326",
            )

        return fill_scene_defaults(d)

    @staticmethod
    def _dimap_times(elements):
        """Parse the ``TIME`` child of each element into a naive datetime index."""
        return pd.to_datetime([el.findtext("TIME").replace("Z", "") for el in elements])

    def getEphem_gdf(self, root):
        """
        Create an ephemeris GeoDataFrame from a parsed DIMAP document.

        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Parsed ``Dimap_Document`` root element.

        Returns
        -------
        geopandas.GeoDataFrame
            Time-indexed GeoDataFrame with ``x, y, z`` positions (m) and
            ``dx, dy, dz`` velocities (m/s) in ECEF (EPSG:4978), plus NaN
            ``cov_*`` columns (DIMAP provides no ephemeris covariance).
        """
        points = root.findall(".//Refined_Model/Ephemeris/Point_List/Point")
        pos = np.array(
            [[float(v) for v in pt.findtext("LOCATION_XYZ").split()] for pt in points]
        )
        vel = np.array(
            [[float(v) for v in pt.findtext("VELOCITY_XYZ").split()] for pt in points]
        )
        eph_df = pd.DataFrame(
            np.hstack([pos, vel]), columns=["x", "y", "z", "dx", "dy", "dz"]
        )
        for n in ["11", "12", "13", "22", "23", "33"]:
            eph_df[f"cov_{n}"] = np.nan
        eph_df["time"] = self._dimap_times(points)
        eph_df.set_index("time", inplace=True)
        return gpd.GeoDataFrame(
            eph_df,
            geometry=gpd.points_from_xy(eph_df["x"], eph_df["y"], eph_df["z"]),
            crs="EPSG:4978",
        )

    def getAtt_df(self, root):
        """
        Create an attitude DataFrame from a parsed DIMAP document.

        Pléiades Neo, SPOT 6/7, and PeruSat-1 products tabulate quaternion
        samples in a ``Quaternion_List``; Pléiades 1A/1B products instead
        provide one polynomial per quaternion component
        (``Polynomial_Quaternions``), which is evaluated at the ephemeris
        timestamps so both shapes come out as the same tabulated DataFrame.

        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Parsed ``Dimap_Document`` root element.

        Returns
        -------
        pandas.DataFrame
            Time-indexed DataFrame with scalar-last quaternions ``q1..q4``
            (Airbus ``Q0`` is the scalar part and lands in ``q4``), plus NaN
            ``cov_*`` columns (DIMAP provides no attitude covariance).
        """
        quats = root.findall(".//Refined_Model/Attitudes/Quaternion_List/Quaternion")
        if quats:
            q = np.array(
                [
                    [float(qq.findtext(k)) for k in ("Q1", "Q2", "Q3", "Q0")]
                    for qq in quats
                ]
            )
            times = self._dimap_times(quats)
        else:
            q, times = self._evaluate_polynomial_quaternions(root)
        att_df = pd.DataFrame(q, columns=["q1", "q2", "q3", "q4"])
        for n in ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]:
            att_df[f"cov_{n}"] = np.nan
        att_df["time"] = times
        att_df.set_index("time", inplace=True)
        return att_df

    @classmethod
    def _evaluate_polynomial_quaternions(cls, root):
        """Evaluate Pléiades 1A/1B polynomial attitude at ephemeris times.

        1A/1B products express each quaternion component ``Q0..Q3`` as a
        degree-3 polynomial with ascending-power ``COEFFICIENTS``, evaluated
        at the scaled time ``(t - (midnight + OFFSET)) / SCALE``, where
        ``OFFSET`` is in seconds after midnight of the acquisition start date
        (#161). This mirrors ASP's reader (``read_attitudes_1A1B`` in
        PleiadesXML.cc and ``get_camera_pose_at_time`` in
        LinescanPleiadesModel.cc). Sampling at the ephemeris timestamps keeps
        the attitude table aligned with the ephemeris; the evaluated
        quaternions are normalized, since polynomial output is not unit-norm
        by construction.

        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Parsed ``Dimap_Document`` root element.

        Returns
        -------
        tuple of (numpy.ndarray, pandas.DatetimeIndex)
            ``(N, 4)`` array of scalar-last unit quaternions ``q1..q4`` and
            the ``N`` ephemeris timestamps they were evaluated at.
        """
        poly = root.find(".//Refined_Model/Attitudes/Polynomial_Quaternions")
        if poly is None:
            raise ValueError(
                "No attitude found under Refined_Model/Attitudes: expected "
                "a Quaternion_List (Pléiades Neo, SPOT 6/7, PeruSat-1) or "
                "Polynomial_Quaternions (Pléiades 1A/1B)."
            )

        offset = float(poly.findtext("OFFSET"))
        scale = float(poly.findtext("SCALE"))
        # Coefficients per component, scalar first as on disk (Q0 = w).
        coeffs = []
        for tag in ("Q0", "Q1", "Q2", "Q3"):
            comp = poly.find(tag)
            degree = int(comp.findtext("DEGREE"))
            if degree != 3:
                raise ValueError(
                    f"Expecting the degree of the quaternion polynomial to "
                    f"be 3, got {degree} for {tag}."
                )
            coeffs.append([float(v) for v in comp.findtext("COEFFICIENTS").split()])

        points = root.findall(".//Refined_Model/Ephemeris/Point_List/Point")
        times = cls._dimap_times(points)
        start = root.findtext(".//Refined_Model/Time/Time_Range/START")
        midnight = pd.Timestamp(start.replace("Z", "")).normalize()
        seconds_since_midnight = (times - midnight).total_seconds().to_numpy()
        scaled_t = (seconds_since_midnight - offset) / scale

        # polyval over an ascending-power coefficient matrix: one column per
        # component (w, x, y, z), evaluated at every scaled timestamp.
        wxyz = np.polynomial.polynomial.polyval(scaled_t, np.array(coeffs).T)
        q = np.column_stack([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return q, times
