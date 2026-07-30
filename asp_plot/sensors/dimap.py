"""Metadata reader for Airbus DIMAP v2 camera files (Pléiades / Pléiades Neo)."""

import os
import xml.etree.ElementTree as ET
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon

from asp_plot.sensors.base import SensorMetadata, _common_base
from asp_plot.utils import glob_file


class PleiadesMetadata(SensorMetadata):
    """Metadata reader for Airbus Pléiades / Pléiades Neo DIMAP camera files.

    Parses DIMAP v2 primary-product metadata (``DIM_*.XML``, root tag
    ``Dimap_Document``) as delivered with Pléiades 1A/1B and Pléiades Neo
    SEN(sor) products. Each scene is delivered as a single DIM XML, so unlike
    WorldView there is no tile mosaicking step. The sidecar ``RPC_*.XML`` files
    share the DIMAP root but carry no ephemeris, attitude, or acquisition-angle
    information, so discovery keeps only files whose ``METADATA_SUBPROFILE`` is
    ``PRODUCT``.

    Notes
    -----
    - Airbus quaternions are scalar-first (``Q0`` is the scalar part). They are
      reordered to the scalar-last ``q1..q4`` layout shared with WorldView, as
      consumed by the roll/pitch/yaw computation in
      :meth:`asp_plot.stereo_geometry.StereoGeometryPlotter._compute_roll_pitch_yaw`.
    - DIMAP reports no ephemeris/attitude covariance and no scan direction or
      TDI level: the ``cov_*`` columns are filled with NaN and ``scandir`` /
      ``tdi`` are None. Consumers treat those as "not provided".
    - The mean view/sun angles and GSD are averaged over the nine
      ``Located_Geometric_Values`` blocks (corners, edge midpoints, center).
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
            self.image_list = self._filter_dimap_products(image_list)
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
    def _is_dimap_product(xml_fn):
        """True if ``xml_fn`` is a DIMAP *product* metadata file.

        A camera scene's metadata is the DIMAP file whose
        ``METADATA_SUBPROFILE`` is ``PRODUCT`` (the ``DIM_*.XML``). The RPC
        sidecars (subprofile ``RPC``) and any non-DIMAP XML are rejected. Uses
        ``iterparse`` and stops at the subprofile tag near the top of the file,
        so detection stays cheap even though DIM files run to several MB.
        """
        try:
            root_seen = False
            for event, el in ET.iterparse(xml_fn, events=("start", "end")):
                if not root_seen:
                    if el.tag != "Dimap_Document":
                        return False
                    root_seen = True
                elif event == "end" and el.tag == "METADATA_SUBPROFILE":
                    return (el.text or "").strip() == "PRODUCT"
            return False
        except (ET.ParseError, OSError):
            return False

    @classmethod
    def _filter_dimap_products(cls, image_list):
        """Keep only DIMAP product XMLs (drops RPC/LUT/index files)."""
        return sorted(f for f in (image_list or []) if cls._is_dimap_product(f))

    @classmethod
    def _discover_xmls(cls, directory, recursive=True):
        """Glob DIMAP product XML files in ``directory``.

        Same shallow-first strategy as the WorldView reader: a flat processing
        directory keeps its camera XMLs at the top level, and only when none
        are found there does discovery recurse into subdirectories (e.g. a raw
        Airbus delivery with ``IMG_01_PNEO3_PAN/DIM_PNEO3_*.XML`` several
        levels deep).
        """
        found = cls._filter_dimap_products(
            glob_file(directory, "*.[Xx][Mm][Ll]", all_files=True)
        )
        if not found and recursive:
            found = cls._filter_dimap_products(
                glob_file(
                    directory, "**/*.[Xx][Mm][Ll]", all_files=True, recursive=True
                )
            )
        return found

    @classmethod
    def detect(cls, directory, recursive=True):
        """Return True if DIMAP product XML files are present."""
        return bool(
            cls._discover_xmls(os.path.expanduser(directory), recursive=recursive)
        )

    @classmethod
    def detect_files(cls, image_list):
        """Return True if any file in ``image_list`` is a DIMAP product XML."""
        return bool(cls._filter_dimap_products(image_list))

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

        mission = root.findtext(".//Strip_Source/MISSION") or "Pleiades"
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

        return d

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
        q = np.array(
            [[float(qq.findtext(k)) for k in ("Q1", "Q2", "Q3", "Q0")] for qq in quats]
        )
        att_df = pd.DataFrame(q, columns=["q1", "q2", "q3", "q4"])
        for n in ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]:
            att_df[f"cov_{n}"] = np.nan
        att_df["time"] = self._dimap_times(quats)
        att_df.set_index("time", inplace=True)
        return att_df
