"""Metadata readers for DIMAP v1-family camera files (SPOT 5, ALOS PRISM).

Both formats predate the DIMAP v2 layout that :mod:`asp_plot.sensors.dimap`
handles, and they share the same v1 skeleton: a ``Metadata_Id`` header (v2 uses
``Metadata_Identification``), scene corners in ``Dataset_Frame``, and a
``Data_Strip`` holding ``Ephemeris/Points/Point`` samples with ``Location`` /
``Velocity`` ``X/Y/Z`` children. They differ in the two places this module
splits on: how the file identifies itself, and where the attitude lives.

The reference implementations mirrored here are ASP's ``SPOT_XML.cc`` (with
``LinescanSpotModel.cc`` for the frame conventions) and ``PRISM_XML.cc`` (with
``prism2asp.cc``).

Unlike the DIMAP v2 and DigitalGlobe products, neither format reports
attitude as quaternions: both tabulate roll/pitch/yaw directly, which is why
``att_df`` carries ``roll``/``pitch``/``yaw`` columns here instead of
``q1..q4`` (see :meth:`SensorMetadata.getAtt_df` consumers, notably
:meth:`asp_plot.stereo_geometry.StereoGeometryPlotter.satellite_position_orientation_plot`).

Both readers are implemented from ASP's reader spec and have not been
validated against a real delivery yet (#168, #179): parsing a scene emits a
one-time warning asking for reports.
"""

import logging
import os
import re
import xml.etree.ElementTree as ET

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon

from asp_plot.sensors.base import (
    SensorMetadata,
    fill_scene_defaults,
    resolve_input_files,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Header tags used to identify a DIMAP v1 file, all of which sit above the bulk
# data blocks. The scan stops at ``Data_Strip``, whose ephemeris, attitude and
# look-angle tables are what make these files large.
_V1_HEADER_TAGS = ("METADATA_PROFILE", "MISSION", "MISSION_INDEX", "INSTRUMENT")
_V1_HEADER_STOP_TAG = "Data_Strip"

# Readers implemented from the ASP spec but not yet validated against a real
# delivery warn once per class (not per file: it is a property of the reader,
# matching the SPEC_ONLY_DIMAP_PROFILES handling in sensors/dimap.py).
_warned_spec_only = set()

# SPOT deliveries this reader claims. ASP's SPOT support is SPOT 5 only
# (``-t spot5``, ``load_spot5_csm_camera_model_from_xml``); the earlier SPOT
# missions share the DIMAP v1 layout but are not ASP stereo sessions, so a
# SPOT 1-4 file is skipped with a warning naming the reason rather than
# silently parsed. SPOT 6/7 are DIMAP *v2* and belong to sensors/dimap.py.
SUPPORTED_SPOT_MISSION_INDEXES = ("5",)
_warned_unsupported_spot = set()  # keyed by absolute path


def _scan_v1_header(xml_fn):
    """Return the DIMAP v1 header tags of ``xml_fn`` as a dict, or None.

    Reads with ``iterparse`` and stops at the first ``Data_Strip`` (or once
    every wanted tag is seen), so identifying a file stays cheap even though
    DIMAP v1 products run to several MB of look-angle and ephemeris tables.

    Returns None for unparseable XML, for non-DIMAP documents, and for DIMAP
    *v2* products: v1 files carry a ``Metadata_Id`` header, v2 files a
    ``Metadata_Identification`` one, and that single tag is what keeps the two
    families' readers from claiming (or warning about) each other's files.
    """
    try:
        found = {}
        root_seen = False
        is_v1 = False
        for event, el in ET.iterparse(xml_fn, events=("start", "end")):
            if not root_seen:
                if el.tag != "Dimap_Document":
                    return None
                root_seen = True
            elif event == "start":
                if el.tag == "Metadata_Id":
                    is_v1 = True
                elif el.tag == _V1_HEADER_STOP_TAG:
                    break
            elif el.tag in _V1_HEADER_TAGS:
                found.setdefault(el.tag, (el.text or "").strip())
                if len(found) == len(_V1_HEADER_TAGS):
                    break
        return found if is_v1 else None
    except (ET.ParseError, OSError):
        return None


def _text(root, path):
    """Return the stripped text at ``path``, or None when the tag is absent."""
    value = root.findtext(path)
    return value.strip() if value is not None else None


def _float_or_nan(root, path):
    """Return the value at ``path`` as a float, or NaN when absent/unparseable.

    The optional summary fields (view and sun angles, GSD) vary across DIMAP v1
    processing levels and vendors, so a missing tag degrades to the documented
    "not provided" NaN instead of raising (#163).
    """
    value = _text(root, path)
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _sanitize(name):
    """Collapse whitespace in an identifier so it is filename-safe.

    ``DATASET_NAME`` in DIMAP v1 is a human-readable string with spaces, but
    scene ids end up in per-pair figure filenames.
    """
    return re.sub(r"\s+", "_", name.strip())


class DimapV1Metadata(SensorMetadata):
    """Shared base for the DIMAP v1-family readers (SPOT 5, ALOS PRISM).

    Holds everything the two formats share — file discovery from a directory
    or an explicit file list, the ``Dataset_Frame`` footprint, the
    ``Ephemeris/Points`` trajectory, and the roll/pitch/yaw ``att_df``
    assembly. Subclasses supply the identification predicate
    (:meth:`_is_camera_file`), the attitude block location
    (:meth:`getAtt_df`), and the identity/summary fields
    (:meth:`get_scene_dict`).

    Attributes
    ----------
    rpy_frame : str
        Human-readable name of the frame the vendor's roll/pitch/yaw angles
        are expressed in, attached to ``att_df.attrs["rpy_frame"]`` and used
        to label the orientation plot. The two sensors do *not* share one
        (see the subclasses).
    """

    #: Overridden by subclasses; see :attr:`rpy_frame`.
    rpy_frame = "vendor-reported"

    #: Attitude covariance column names, filled with NaN. DIMAP v1 reports no
    #: covariance, but the columns must exist so consumers can test for
    #: "provided" uniformly rather than by sensor.
    _ATT_COV_COLS = ("11", "12", "13", "14", "22", "23", "24", "33", "34", "44")
    _EPH_COV_COLS = ("11", "12", "13", "22", "23", "33")

    def __init__(self, directory=None, image_list=None):
        """
        Initialize a DIMAP v1 metadata reader.

        The reader can be built either from a ``directory`` (its DIMAP v1 XMLs
        are discovered) or from an explicit ``image_list`` of XML files. At
        least one of the two must be given; files the sensor's content check
        does not claim are filtered out in both cases.

        Parameters
        ----------
        directory : str, optional
            Path to directory containing DIMAP v1 camera metadata files.
        image_list : list of str, optional
            Explicit list of XML files to use instead of discovering them
            from ``directory``.

        Raises
        ------
        ValueError
            If neither ``directory`` nor ``image_list`` is given, or if no
            metadata files for this sensor are found.
        """
        self.directory, self.image_list = resolve_input_files(
            type(self), directory, image_list
        )
        if not self.image_list:
            raise ValueError(
                f"\n\nMissing {self.name} DIMAP v1 camera metadata files. "
                "Cannot extract metadata without these.\n\n"
            )

    @classmethod
    def _warn_spec_only(cls):
        """Warn once that this reader is unvalidated against real deliveries."""
        if cls.name in _warned_spec_only:
            return
        _warned_spec_only.add(cls.name)
        logger.warning(
            "%s metadata support is implemented from the ASP reader spec "
            "but not yet validated against real data — please report issues "
            "at https://github.com/uw-cryo/asp_plot/issues/new",
            cls.name,
        )

    def get_scene_dicts(self):
        """Return one sensor-agnostic scene dict per metadata XML."""
        return [self.get_scene_dict(xml) for xml in self.image_list]

    @staticmethod
    def _frame_geom(root):
        """Build the scene footprint from the ``Dataset_Frame`` corners.

        DIMAP v1 stores the four scene corners as ``Vertex`` elements with
        ``FRAME_LON``/``FRAME_LAT`` (plus the ``FRAME_ROW``/``FRAME_COL``
        pixel coordinates ASP uses for the RPC fit). Selection is by tag
        name, which is what skips the ``Center`` element some products add
        with the same children; ASP instead takes the first four children of
        ``Dataset_Frame`` positionally (``SpotXML::read_corners``), whatever
        they are named.

        Neither reader has been validated against a real delivery, so a
        corner count other than the expected four is reported rather than
        quietly turned into a differently-shaped polygon — that is the signal
        a product nests its corners under tags this does not anticipate.
        """
        frame = root.find(".//Dataset_Frame")
        vertices = frame.findall("Vertex") if frame is not None else []
        corners = [
            (float(v.findtext("FRAME_LON")), float(v.findtext("FRAME_LAT")))
            for v in vertices
            if v.findtext("FRAME_LON") is not None
            and v.findtext("FRAME_LAT") is not None
        ]
        if len(corners) < 3:
            raise ValueError(
                "Could not read a scene footprint: expected at least three "
                "Dataset_Frame/Vertex elements with FRAME_LON and FRAME_LAT."
            )
        if len(corners) != 4:
            logger.warning(
                "Dataset_Frame has %d corner vertices, expected 4 — the scene "
                "footprint is built from all of them. Please report this "
                "product at https://github.com/uw-cryo/asp_plot/issues/new",
                len(corners),
            )
        return Polygon(corners)

    @staticmethod
    def _v1_times(elements):
        """Parse the ``TIME`` child of each element into a naive datetime index."""
        return pd.to_datetime([el.findtext("TIME").replace("Z", "") for el in elements])

    def getEphem_gdf(self, root):
        """
        Create an ephemeris GeoDataFrame from a parsed DIMAP v1 document.

        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Parsed ``Dimap_Document`` root element.

        Returns
        -------
        geopandas.GeoDataFrame
            Time-indexed GeoDataFrame with ``x, y, z`` positions (m) and
            ``dx, dy, dz`` velocities (m/s) in ECEF (EPSG:4978), plus NaN
            ``cov_*`` columns (DIMAP v1 provides no ephemeris covariance).
        """
        points = root.findall(".//Ephemeris/Points/Point")
        if not points:
            raise ValueError(
                "No ephemeris found: expected Ephemeris/Points/Point elements."
            )

        def xyz(point, tag):
            node = point.find(tag)
            return [float(node.findtext(k)) for k in ("X", "Y", "Z")]

        pos = np.array([xyz(pt, "Location") for pt in points])
        vel = np.array([xyz(pt, "Velocity") for pt in points])
        eph_df = pd.DataFrame(
            np.hstack([pos, vel]), columns=["x", "y", "z", "dx", "dy", "dz"]
        )
        for n in self._EPH_COV_COLS:
            eph_df[f"cov_{n}"] = np.nan
        eph_df["time"] = self._v1_times(points)
        eph_df.set_index("time", inplace=True)
        return gpd.GeoDataFrame(
            eph_df,
            geometry=gpd.points_from_xy(eph_df["x"], eph_df["y"], eph_df["z"]),
            crs="EPSG:4978",
        )

    def _rpy_att_df(self, times, roll, pitch, yaw):
        """Assemble the tabulated roll/pitch/yaw attitude DataFrame.

        Parameters
        ----------
        times : array-like of datetime
            Attitude sample times.
        roll, pitch, yaw : array-like of float
            Vendor-reported angles **in degrees** (subclasses convert if the
            format stores radians).

        Returns
        -------
        pandas.DataFrame
            Time-indexed DataFrame with ``roll``, ``pitch``, ``yaw`` (degrees)
            and NaN ``cov_*`` columns. ``attrs["rpy_frame"]`` names the frame
            the angles are expressed in, since it is sensor-specific and
            plots must not imply the values are cross-sensor comparable.
        """
        att_df = pd.DataFrame({"roll": roll, "pitch": pitch, "yaw": yaw})
        for n in self._ATT_COV_COLS:
            att_df[f"cov_{n}"] = np.nan
        att_df["time"] = times
        att_df.set_index("time", inplace=True)
        att_df.attrs["rpy_frame"] = self.rpy_frame
        return att_df


class Spot5Metadata(DimapV1Metadata):
    """Metadata reader for SPOT 5 DIMAP v1 scene metadata.

    Parses the ``METADATA.DIM`` / ``*.XML`` scene metadata delivered with SPOT
    5 HRG/HRS products, mirroring ASP's ``SPOT_XML.cc`` (the ``spot5``
    session). Ephemeris and attitude both live in ``Data_Strip``; the scene
    footprint comes from ``Dataset_Frame``.

    Notes
    -----
    - Attitude is tabulated as yaw/pitch/roll **in radians** under
      ``Corrected_Attitudes/Corrected_Attitude/Angles`` and converted to
      degrees here. ASP feeds the raw values straight to ``sin``/``cos`` in
      ``get_look_rotation_matrix``, which is what fixes the unit.
    - The angles are defined in the SPOT 123-4-5 Geometry Handbook navigation
      frame, relative to *its* local orbital frame (X across-track, Y
      along-track, Z up), composed as ``Mp*Mr*My``. That is a different axis
      assignment from the (along, across, down) frame the quaternion sensors'
      roll/pitch/yaw are computed in, so the values are reported as delivered
      and labeled with their own frame rather than silently mixed with the
      others.
    - DIMAP v1 reports no satellite azimuth, so ``meansataz`` stays NaN and
      the pair convergence angle degrades to NaN; the incidence, viewing and
      sun angles that *are* in ``Scene_Source`` are filled.
    """

    name = "SPOT5"
    rpy_frame = "SPOT Geometry Handbook navigation frame"

    @classmethod
    def _is_camera_file(cls, path):
        """True if ``path`` is a SPOT 5 DIMAP v1 scene metadata file.

        Identifies on ``MISSION``/``MISSION_INDEX`` in ``Scene_Source``: ASP's
        reader itself is selected by the ``-t spot5`` session and checks no
        profile tag, so those two are the only content markers available.
        A SPOT file from another mission index is rejected with a one-time
        warning naming it, so the user learns why it was skipped.
        """
        header = _scan_v1_header(path)
        if not header or header.get("MISSION", "").upper() != "SPOT":
            return False
        index = header.get("MISSION_INDEX", "")
        if index not in SUPPORTED_SPOT_MISSION_INDEXES:
            key = os.path.abspath(path)
            if key not in _warned_unsupported_spot:
                _warned_unsupported_spot.add(key)
                logger.warning(
                    "SPOT %s scene skipped: %s (only SPOT %s is an ASP stereo "
                    "session; broader sensor support is tracked in "
                    "uw-cryo/asp_plot#168)",
                    index or "?",
                    path,
                    ", ".join(SUPPORTED_SPOT_MISSION_INDEXES),
                )
            return False
        return True

    def get_scene_dict(self, xml, geteph=True):
        """
        Get a dictionary of metadata for one SPOT 5 scene.

        Parameters
        ----------
        xml : str
            Path to the SPOT 5 DIMAP v1 scene metadata file.
        geteph : bool, optional
            Whether to extract ephemeris/attitude data, default is True.

        Returns
        -------
        dict
            Sensor-agnostic scene dict (see the package docstring).
        """
        self._warn_spec_only()
        root = ET.parse(xml).getroot()

        geom = self._frame_geom(root)
        source = ".//Dataset_Sources/Source_Information/Scene_Source/"

        # The scene center time is what ASP reads (and the only full timestamp
        # in the header); IMAGING_DATE/IMAGING_TIME is the fallback.
        center_time = _text(
            root, ".//Sensor_Configuration/Time_Stamp/SCENE_CENTER_TIME"
        )
        if center_time is None:
            center_time = (
                f"{_text(root, source + 'IMAGING_DATE')}T"
                f"{_text(root, source + 'IMAGING_TIME')}"
            )
        date = pd.Timestamp(center_time.replace("Z", "")).to_pydatetime()

        catid = _text(root, ".//Dataset_Id/DATASET_NAME")
        index = _text(root, source + "MISSION_INDEX") or ""

        d = {
            "xml_fn": xml,
            "catid": (
                _sanitize(catid)
                if catid
                else os.path.splitext(os.path.basename(xml))[0]
            ),
            "sensor": f"SPOT{index}",
            "date": date,
            "geom": geom,
            "meansatel": np.round(
                90.0 - _float_or_nan(root, source + "INCIDENCE_ANGLE"), 2
            ),
            "meanoffnadirviewangle": _float_or_nan(root, source + "VIEWING_ANGLE"),
            "meanintrackviewangle": _float_or_nan(
                root, source + "VIEWING_ANGLE_ALONG_TRACK"
            ),
            "meancrosstrackviewangle": _float_or_nan(
                root, source + "VIEWING_ANGLE_ACROSS_TRACK"
            ),
            "meanproductgsd": _float_or_nan(root, source + "THEORETICAL_RESOLUTION"),
            "meansunaz": _float_or_nan(root, source + "SUN_AZIMUTH"),
            "meansunel": _float_or_nan(root, source + "SUN_ELEVATION"),
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

    def getAtt_df(self, root):
        """
        Create an attitude DataFrame from a parsed SPOT 5 document.

        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Parsed ``Dimap_Document`` root element.

        Returns
        -------
        pandas.DataFrame
            Time-indexed DataFrame with ``roll``, ``pitch``, ``yaw`` in
            degrees (converted from the radians on disk) and NaN ``cov_*``
            columns.
        """
        angles = root.findall(".//Corrected_Attitudes/Corrected_Attitude/Angles")
        if not angles:
            raise ValueError(
                "No attitude found: expected Corrected_Attitudes/"
                "Corrected_Attitude/Angles elements with YAW, PITCH and ROLL."
            )
        # Read by tag name, not position: the file lists YAW, PITCH, ROLL in
        # that order, which is the reverse of the column order used here.
        rpy = np.degrees(
            np.array(
                [
                    [float(a.findtext(k)) for k in ("ROLL", "PITCH", "YAW")]
                    for a in angles
                ]
            )
        )
        return self._rpy_att_df(self._v1_times(angles), rpy[:, 0], rpy[:, 1], rpy[:, 2])


class PrismMetadata(DimapV1Metadata):
    """Metadata reader for ALOS PRISM DIMAP-style scene metadata.

    Parses the ALOS PRISM ``*.DIMA``/``*.XML`` metadata that ASP's
    ``PRISM_XML.cc`` reads (used by ``prism2asp`` to build CSM linescan
    cameras). A PRISM triplet delivers one file per view (forward, nadir,
    backward), each of which is one scene here.

    Notes
    -----
    - Attitude is tabulated as roll/pitch/yaw **in degrees** under
      ``Satellite_Attitudes/Angles_List/Angles/Angle``; ASP applies them with
      ``rollPitchYaw()``, which converts degrees to radians.
    - Those angles are the satellite's attitude relative to the nominal
      (along, across, down) orbital frame, composed as ``Rz(yaw) Ry(pitch)
      Rx(roll)`` — the same frame and Euler convention as the roll/pitch/yaw
      this package derives from quaternion sensors' attitude, so PRISM curves
      are directly comparable to those. The fixed per-view mounting rotation
      ASP applies on top (``cam2sat``) is camera geometry, not satellite
      attitude, and is deliberately not folded in here.
    - PRISM reports no view or GSD summary angles in the metadata header, so
      those degrade to NaN; the sun angles are filled when present.
    """

    name = "PRISM"
    rpy_frame = "orbital frame (along, across, down)"

    @classmethod
    def _is_camera_file(cls, path):
        """True if ``path`` is an ALOS PRISM metadata file.

        Gates on ``METADATA_PROFILE == "ALOS"``, exactly as ASP's
        ``parsePrismXml`` does.
        """
        header = _scan_v1_header(path)
        return bool(header) and header.get("METADATA_PROFILE") == "ALOS"

    def get_scene_dict(self, xml, geteph=True):
        """
        Get a dictionary of metadata for one ALOS PRISM scene.

        Parameters
        ----------
        xml : str
            Path to the PRISM DIMAP-style metadata file.
        geteph : bool, optional
            Whether to extract ephemeris/attitude data, default is True.

        Returns
        -------
        dict
            Sensor-agnostic scene dict (see the package docstring).
        """
        self._warn_spec_only()
        root = ET.parse(xml).getroot()

        geom = self._frame_geom(root)
        source = ".//Dataset_Sources/Source_Information/Scene_Source/"

        first_line = _text(root, ".//Satellite_Time/TIME_FIRST_LINE")
        if first_line is None:
            first_line = (
                f"{_text(root, source + 'IMAGING_DATE')}T"
                f"{_text(root, source + 'IMAGING_TIME')}"
            )
        date = pd.Timestamp(first_line.replace("Z", "")).to_pydatetime()

        catid = _text(root, ".//Dataset_Id/DATASET_NAME")
        mission = _text(root, source + "MISSION") or "ALOS"
        # INSTRUMENT carries the view (forward / nadir / backward), which is
        # what distinguishes the scenes of a PRISM triplet.
        instrument = _text(root, source + "INSTRUMENT") or "PRISM"

        d = {
            "xml_fn": xml,
            "catid": (
                _sanitize(catid)
                if catid
                else os.path.splitext(os.path.basename(xml))[0]
            ),
            "sensor": _sanitize(f"{mission}-{instrument}"),
            "date": date,
            "geom": geom,
            "meansunaz": _float_or_nan(root, source + "SUN_AZIMUTH"),
            "meansunel": _float_or_nan(root, source + "SUN_ELEVATION"),
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

    def getAtt_df(self, root):
        """
        Create an attitude DataFrame from a parsed PRISM document.

        Parameters
        ----------
        root : xml.etree.ElementTree.Element
            Parsed ``Dimap_Document`` root element.

        Returns
        -------
        pandas.DataFrame
            Time-indexed DataFrame with ``roll``, ``pitch``, ``yaw`` in
            degrees (as delivered) and NaN ``cov_*`` columns.
        """
        angles = root.findall(".//Satellite_Attitudes/Angles_List/Angles")
        if not angles:
            raise ValueError(
                "No attitude found: expected Satellite_Attitudes/Angles_List/"
                "Angles elements with an Angle child holding ROLL, PITCH, YAW."
            )
        rpy = np.array(
            [
                [float(a.find("Angle").findtext(k)) for k in ("ROLL", "PITCH", "YAW")]
                for a in angles
            ]
        )
        return self._rpy_att_df(self._v1_times(angles), rpy[:, 0], rpy[:, 1], rpy[:, 2])
