"""Sensor-specific metadata readers for stereo scenes.

This module isolates the *sensor-specific* work of discovering scene files and
extracting per-scene metadata from the *sensor-agnostic* stereo-pair geometry
math in :mod:`asp_plot.stereopair_metadata_parser`.

The goal is flexibility: today only WorldView (and other DigitalGlobe-heritage)
XML camera files are supported, but adding a new sensor (ASTER, HiRISE, etc.) is
a matter of writing a new :class:`SensorMetadata` subclass and registering it in
``SENSORS`` — no changes to the pair-level geometry code are required.

Each reader is responsible for turning a directory of camera/metadata files into
a list of *scene dicts*, one per scene, each containing the sensor-agnostic keys
the geometry code consumes:

``xml_fn``, ``catid``, ``sensor``, ``date``, ``scandir``, ``tdi``, ``geom``
(a Shapely polygon footprint in EPSG:4326), the mean view-angle/GSD/sun
attributes (``meansataz``, ``meansatel``, ``meanoffnadirviewangle``,
``meanintrackviewangle``, ``meancrosstrackviewangle``, ``meanproductgsd``,
``meansunaz``, ``meansunel``, ``cloudcover``), and — when ``geteph`` is True —
``eph_gdf`` (ephemeris GeoDataFrame in EPSG:4978), ``att_df`` (attitude
DataFrame), and ``fp_gdf`` (footprint GeoDataFrame in EPSG:4326).
"""

import glob
import logging
import os
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import Polygon, union_all, wkt

from asp_plot.utils import get_xml_tag, glob_file, run_subprocess_command

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SensorMetadata(ABC):
    """Abstract base class for a single sensor's metadata reader.

    A concrete reader discovers the scene files for one sensor in a directory and
    extracts a list of sensor-agnostic *scene dicts* (see the module docstring
    for the schema) that the stereo-pair geometry code can consume without
    knowing which sensor produced them.

    Subclasses must implement :meth:`detect` (so the sensor can be chosen
    automatically) and :meth:`get_scene_dicts`.

    Attributes
    ----------
    name : str
        Human-readable sensor name (e.g. ``"WorldView"``).
    directory : str
        Path to the directory containing the sensor's metadata files.
    """

    name = "sensor"

    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing the sensor's camera/metadata files.
        """
        self.directory = os.path.expanduser(directory)

    @classmethod
    @abstractmethod
    def detect(cls, directory, recursive=True):
        """Return True if this reader can handle the files in ``directory``.

        Parameters
        ----------
        directory : str
            Path to directory to inspect.
        recursive : bool, optional
            If True (default), also match metadata files nested in
            subdirectories. :func:`sensor_for_directory` first asks every
            sensor to detect shallowly and only then recursively, so a sensor
            matching at the top level wins over one matching a nested delivery.

        Returns
        -------
        bool
            Whether this sensor's metadata files are present.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def detect_files(cls, image_list):
        """Return True if this reader can handle the files in ``image_list``.

        The file-list counterpart of :meth:`detect`, used when the sensor must
        be chosen from an explicit list of inputs rather than a directory.

        Parameters
        ----------
        image_list : list of str
            Candidate metadata file paths.

        Returns
        -------
        bool
            Whether this sensor's metadata files are present in the list.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scene_dicts(self):
        """Return a list of per-scene metadata dictionaries.

        Returns
        -------
        list of dict
            One sensor-agnostic scene dict per scene (see module docstring).
        """
        raise NotImplementedError


class WorldViewMetadata(SensorMetadata):
    """Metadata reader for WorldView satellite XML camera files.

    Parses WorldView (and other DigitalGlobe-heritage products that share the
    same XML format, e.g. GeoEye-1, QuickBird, IKONOS) satellite XML files to
    extract per-scene metadata, handling both single XML files and multiple XML
    tiles per scene (mosaicked with ``dg_mosaic``).

    This class is named for the *sensor family* (the stable WorldView name) and
    governs which reader parses the XML. It is intentionally distinct from the
    *attribution* check :func:`asp_plot.utils.detect_vantor_satellite`, which is
    named for the rights-holder (Vantor) and decides whether the "© Vantor"
    overlay applies. The two concerns use different names on purpose; see #137.

    Attributes
    ----------
    directory : str
        Path to directory containing XML files.
    image_list : list
        List of XML files found in the directory.
    """

    name = "WorldView"

    def __init__(self, directory=None, image_list=None):
        """
        Initialize the WorldView metadata reader.

        The reader can be built either from a ``directory`` (its camera XMLs are
        discovered) or from an explicit ``image_list`` of XML files (e.g. a
        ``geom_plot *.XML`` invocation, where the shell has already expanded the
        files). At least one of the two must be given.

        Parameters
        ----------
        directory : str, optional
            Path to directory containing XML camera model files. When
            ``image_list`` is also given, this is used only as the base
            directory for ``dg_mosaic`` outputs and the pair name.
        image_list : list of str, optional
            Explicit list of XML camera files to use instead of discovering
            them from ``directory``. Non-camera XMLs (``README.XML``,
            ``*ortho*.xml``) are still filtered out.

        Raises
        ------
        ValueError
            If neither ``directory`` nor ``image_list`` is given, or if no
            camera XML files are found.
        """
        if directory is None and image_list is None:
            raise ValueError("Provide either a directory or an image_list.")

        if image_list is not None:
            # Explicit file list (e.g. shell-expanded ``geom_plot *.XML``): use
            # it directly, but still drop non-camera XMLs by name. Fall back to
            # the files' common parent for mosaic output / pair naming when no
            # directory is supplied.
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
                "\n\nMissing XML camera files. Cannot extract metadata without these.\n\n"
            )

    # XML files that are delivered alongside the camera models but are *not*
    # camera models themselves and must be ignored. Matched against the file
    # basename, case-insensitively: ortho products (``*ortho*.xml``) and the
    # ``README.XML`` that ships in every DigitalGlobe-heritage delivery.
    _NON_CAMERA_XML_RE = re.compile(r"ortho|readme", re.IGNORECASE)

    @staticmethod
    def _filter_camera_xmls(image_list):
        """Drop non-camera XMLs (ortho products, README.XML, ...) by basename.

        Matched against the basename so a parent directory named e.g.
        ``ortho_run/`` does not exclude otherwise-valid scenes.
        """
        return sorted(
            file
            for file in (image_list or [])
            if not WorldViewMetadata._NON_CAMERA_XML_RE.search(os.path.basename(file))
        )

    @staticmethod
    def _discover_xmls(directory, recursive=True):
        """Glob camera-model XML files in ``directory``.

        Searches the top level of ``directory`` first: a flat delivery or an ASP
        processing directory keeps its camera XMLs there, so those take
        precedence and unrelated XMLs in subdirectories are not pulled in. Only
        when the top level has no camera XML does it fall back to a recursive
        search, because some satellite deliveries nest the camera XML several
        subdirectories deep (e.g. ``.../<order>/DVD_VOL_1/<order>/<scene>_PAN/
        <scene>.XML``) alongside ``README.XML`` files that must be ignored. This
        lets a user point at such a delivery without flattening it first, while
        leaving the behavior of flat directories unchanged.

        Parameters
        ----------
        directory : str
            Path to directory to search.
        recursive : bool, optional
            If True (default), fall back to a recursive search when the top
            level holds no camera XML. If False, only the top level is searched.

        Returns
        -------
        list
            Sorted list of XML file paths, excluding non-camera XMLs such as
            ``*ortho*.xml`` and ``README.XML``.
        """
        # glob_file returns None (not []) when nothing matches.
        found = WorldViewMetadata._filter_camera_xmls(
            glob_file(directory, "*.[Xx][Mm][Ll]", all_files=True)
        )
        if not found and recursive:
            found = WorldViewMetadata._filter_camera_xmls(
                glob_file(
                    directory, "**/*.[Xx][Mm][Ll]", all_files=True, recursive=True
                )
            )
        return found

    @classmethod
    def detect(cls, directory, recursive=True):
        """Return True if non-ortho XML camera files are present.

        Parameters
        ----------
        directory : str
            Path to directory to inspect.
        recursive : bool, optional
            If True (default), fall back to searching subdirectories when the
            top level has no camera XML.

        Returns
        -------
        bool
            Whether WorldView XML camera files were found.
        """
        return bool(
            cls._discover_xmls(os.path.expanduser(directory), recursive=recursive)
        )

    @classmethod
    def detect_files(cls, image_list):
        """Return True if any file in ``image_list`` is a camera XML.

        The file-list counterpart of :meth:`detect`, used to choose a reader
        for an explicit list of inputs (see :func:`sensor_for_inputs`). A file
        is a camera XML if it survives the non-camera basename filter (so
        ``README.XML`` / ``*ortho*.xml`` alone do not match).

        Parameters
        ----------
        image_list : list of str
            Candidate XML file paths.

        Returns
        -------
        bool
            Whether any camera XML files are present.
        """
        return bool(cls._filter_camera_xmls(image_list))

    def get_scene_dicts(self):
        """
        Get dictionaries of metadata for each catalog ID.

        Builds dictionaries of metadata for each catalog ID found in the XML files.

        Returns
        -------
        list
            List of dictionaries, one for each catalog ID, containing metadata
        """
        catid_xmls = self.get_catid_xmls()
        catid_dicts = []
        for catid, xml in catid_xmls.items():
            catid_dicts.append(self.get_id_dict(catid, xml))
        return catid_dicts

    @staticmethod
    def _read_catid(xml_file):
        """Return the CATID of an XML file, or None if it has none.

        A delivery may contain XML files that are not camera models (e.g. a
        stray metadata sidecar) and therefore carry no ``CATID`` tag. Those are
        not camera scenes and should be skipped rather than crashing discovery,
        so this swallows the missing-tag/parse errors and returns None.

        Parameters
        ----------
        xml_file : str
            Path to the XML file.

        Returns
        -------
        str or None
            The catalog ID, or None if the file has no ``CATID`` tag or cannot
            be parsed as XML.
        """
        try:
            return get_xml_tag(xml_file, "CATID")
        except (ValueError, ET.ParseError):
            return None

    def get_catid_xmls(self):
        """
        Get a single representative XML file for each catalog ID.

        Groups the discovered XML files by their catalog ID (read from the XML
        content, not the filename) and resolves each scene to one XML: a scene
        delivered as a single XML is used as-is, while a scene tiled across
        multiple XMLs is mosaicked into one with ``dg_mosaic``.

        Returns
        -------
        dict
            Dictionary mapping catalog IDs to a single XML file path.

        Raises
        ------
        ValueError
            If none of the discovered XML files contain a ``CATID`` tag.

        Notes
        -----
        Mosaicking is decided per catalog ID, so a directory holding many
        distinct single-tile scenes is *not* mosaicked just because it contains
        more than two XML files. Mosaicking a tiled scene requires ``dg_mosaic``
        from the NASA Ames Stereo Pipeline on the system path.
        """
        # Group every discovered XML by CATID read from XML content (filenames
        # are not reliable). Files without a CATID are not camera models (e.g. a
        # README.XML that slipped past the name filter) and are skipped with a
        # warning rather than crashing.
        catid_groups = {}
        for xml_file in self.image_list:
            catid = self._read_catid(xml_file)
            if catid is None:
                logger.warning(
                    "Skipping XML without a CATID tag (not a camera model): %s",
                    xml_file,
                )
                continue
            catid_groups.setdefault(catid, []).append(xml_file)

        if not catid_groups:
            raise ValueError(
                "\n\nNo XML camera files with a CATID tag found in directory.\n\n"
            )

        # Resolve each CATID to a single representative XML. A mosaic output
        # (``*.r100.xml`` / ``*.r50.xml``) is only a regenerable intermediate
        # when raw tiles for the same CATID are also present; when it is the
        # only XML for a CATID it *is* the delivered camera and is used as-is.
        catid_xmls = {}
        for catid, group in sorted(catid_groups.items()):
            raw_tiles = sorted(
                f for f in group if not re.search(r"\.r100\.|\.r50\.", f)
            )
            if not raw_tiles:
                # Delivered as a single, already-mosaicked XML (e.g. *.r100.xml).
                catid_xmls[catid] = sorted(group)[0]
            elif len(raw_tiles) == 1:
                # Single tile: use it directly, no mosaicking needed.
                catid_xmls[catid] = raw_tiles[0]
            else:
                print(
                    f"\nCATID {catid} is tiled across {len(raw_tiles)} XMLs. "
                    "Mosaicking before proceeding.\n"
                )
                catid_xmls[catid] = self._mosaic_tiles(catid, raw_tiles)

        return catid_xmls

    def _mosaic_tiles(self, catid, tile_xmls):
        """
        Mosaic the tile XMLs of a single catalog ID into one XML.

        Uses ``dg_mosaic`` to merge the image tiles of one scene into a single
        camera XML. An existing mosaic output is reused rather than regenerated.

        Parameters
        ----------
        catid : str
            Catalog ID the tiles belong to (used for the output filename).
        tile_xmls : list of str
            Paths to the tile XML files for this catalog ID.

        Returns
        -------
        str
            Path to the mosaicked ``*.r100.xml`` file.

        Notes
        -----
        Requires dg_mosaic from the NASA Ames Stereo Pipeline to be installed
        and available in the system path.
        """
        output_xml = os.path.join(self.directory, f"{catid}_asp_plot_dg_mosaic")
        output_xml_r100 = f"{output_xml}.r100.xml"

        if not os.path.exists(output_xml_r100):
            # Build the command string instead of a list, needed for subprocess call, .split() below
            xml_files = " ".join(tile_xmls)
            command = (
                f"dg_mosaic --skip-tif-gen --output-prefix {output_xml} {xml_files}"
            )

            print(f"\nRunning dg_mosaic with command: {command}\n")

            # Run the command
            run_subprocess_command(command.split())
        else:
            print(f"\nUsing existing mosaicked XML file: {output_xml_r100}\n")

        return output_xml_r100

    def get_id_dict(self, catid, xml, geteph=True):
        """
        Get a dictionary of metadata for a specific catalog ID.

        Extracts metadata from XML file for a given catalog ID, including
        satellite parameters, acquisition angles, and geometry.

        Parameters
        ----------
        catid : str
            Catalog ID for the satellite image
        xml : str
            Path to the XML file
        geteph : bool, optional
            Whether to extract ephemeris data, default is True

        Returns
        -------
        dict
            Dictionary containing metadata for the catalog ID

        Notes
        -----
        The dictionary includes satellite ID, acquisition date, scan direction,
        TDI level, geometry information, and various mean angles and parameters.
        If geteph is True, also includes ephemeris and footprint GeoDataFrames.
        """

        def list_average(list):
            """Calculate average of values in a list, handling NaN values."""
            return np.round(pd.Series(list, dtype=float).dropna().mean(), 2)

        attributes = {
            "MEANSATAZ": [],
            "MEANSATEL": [],
            "MEANOFFNADIRVIEWANGLE": [],
            "MEANINTRACKVIEWANGLE": [],
            "MEANCROSSTRACKVIEWANGLE": [],
            "MEANPRODUCTGSD": [],
            "MEANSUNAZ": [],
            "MEANSUNEL": [],
            "CLOUDCOVER": [],
            "geom": [],
        }

        for tag, lst in attributes.items():
            if tag != "geom":
                lst.append(get_xml_tag(xml, tag))
            else:
                # This returns a Shapely Polygon geometry
                lst.append(self.xml2poly(xml))

        d = {
            "xml_fn": xml,
            "catid": catid,
            "sensor": get_xml_tag(xml, "SATID"),
            "date": datetime.strptime(
                get_xml_tag(xml, "FIRSTLINETIME"), "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "scandir": get_xml_tag(xml, "SCANDIRECTION"),
            "tdi": int(get_xml_tag(xml, "TDILEVEL")),
            "geom": union_all(attributes["geom"]),
        }

        # Add Ephemeris GeoDataFrame, Attitude DataFrame, and Footprint GeoDataFrame
        if geteph:
            d["eph_gdf"] = self.getEphem_gdf(xml)
            d["att_df"] = self.getAtt_df(xml)
            d["fp_gdf"] = gpd.GeoDataFrame(
                {"idx": [0], "geometry": d["geom"]},
                geometry="geometry",
                crs="EPSG:4326",
            )

        # Compute mean values when multiple xml make up a single image ID
        for tag, lst in attributes.items():
            if tag != "geom":
                d[tag.lower()] = list_average(lst)

        return d

    def getEphem(self, xml):
        """
        Extract ephemeris data from XML file.

        Retrieves satellite ephemeris (position and velocity) data from the XML file.

        Parameters
        ----------
        xml : str
            Path to the XML file

        Returns
        -------
        numpy.ndarray
            Array containing ephemeris data with columns:
            point_num, Xpos, Ypos, Zpos, Xvel, Yvel, Zvel, and covariance matrix elements

        Notes
        -----
        All coordinates are in Earth-Centered Fixed (ECF) reference frame.
        Units are meters for positions, meters/sec for velocities, and m^2 for covariance.
        """
        e = get_xml_tag(xml, "EPHEMLIST", all=True)
        # Could get fancy with structured array here
        # point_num, Xpos, Ypos, Zpos, Xvel, Yvel, Zvel, covariance matrix (6 elements)
        # dtype=[('point', 'i4'), ('Xpos', 'f8'), ('Ypos', 'f8'), ('Zpos', 'f8'), ('Xvel', 'f8') ...]
        # All coordinates are ECF, meters, meters/sec, m^2
        return np.array([i.split() for i in e], dtype=np.float64)

    def getAtt(self, xml):
        """
        Extract attitude data from XML file.

        Retrieves satellite attitude (orientation quaternion and covariance) data
        from the XML file.

        Parameters
        ----------
        xml : str
            Path to the XML file

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 15) containing attitude data with columns:
            point_num, q1, q2, q3, q4, and 10 covariance matrix elements
            (upper triangle of 4x4 matrix)
        """
        a = get_xml_tag(xml, "ATTLIST", all=True)
        return np.array([i.split() for i in a], dtype=np.float64)

    def getEphem_gdf(self, xml):
        """
        Create a GeoDataFrame from ephemeris data.

        Converts ephemeris data to a GeoDataFrame with time index and Point geometry.

        Parameters
        ----------
        xml : str
            Path to the XML file

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with ephemeris data and Point geometries in EPSG:4978

        Notes
        -----
        The GeoDataFrame uses EPSG:4978 (Earth-Centered Earth-Fixed) CRS and
        has a time index corresponding to the acquisition times.
        """
        names = [
            "index",
        ]
        names.extend(["x", "y", "z"])
        names.extend(["dx", "dy", "dz"])
        names.extend(["cov_{}".format(n) for n in ["11", "12", "13", "22", "23", "33"]])
        e = self.getEphem(xml)
        t0 = pd.to_datetime(get_xml_tag(xml, "STARTTIME"))
        dt = pd.Timedelta(float(get_xml_tag(xml, "TIMEINTERVAL")), unit="s")
        eph_df = pd.DataFrame(e, columns=names)
        eph_df["time"] = t0 + eph_df.index * dt
        eph_df.set_index("time", inplace=True)
        eph_gdf = gpd.GeoDataFrame(
            eph_df,
            geometry=gpd.points_from_xy(eph_df["x"], eph_df["y"], eph_df["z"]),
            crs="EPSG:4978",
        )
        return eph_gdf

    def getAtt_df(self, xml):
        """
        Create a DataFrame from attitude data.

        Converts attitude data to a DataFrame with time index.

        Parameters
        ----------
        xml : str
            Path to the XML file

        Returns
        -------
        pandas.DataFrame
            DataFrame with attitude quaternions and covariance, time-indexed
        """
        names = ["index", "q1", "q2", "q3", "q4"]
        names.extend(
            [
                "cov_{}".format(n)
                for n in ["11", "12", "13", "14", "22", "23", "24", "33", "34", "44"]
            ]
        )
        a = self.getAtt(xml)
        t0 = pd.to_datetime(get_xml_tag(xml, "STARTTIME"))
        dt = pd.Timedelta(float(get_xml_tag(xml, "TIMEINTERVAL")), unit="s")
        att_df = pd.DataFrame(a, columns=names)
        att_df["time"] = t0 + att_df.index * dt
        att_df.set_index("time", inplace=True)
        return att_df

    def xml2wkt(self, xml):
        """
        Convert XML corner coordinates to WKT polygon string.

        Extracts corner coordinates from XML file and converts them to a
        Well-Known Text (WKT) polygon string.

        Parameters
        ----------
        xml : str
            Path to the XML file

        Returns
        -------
        str
            WKT polygon string representation of image footprint

        Notes
        -----
        Uses ULLON/ULLAT, URLON/URLAT, LRLON/LRLAT, LLLON/LLLAT tags
        (Upper-Left, Upper-Right, Lower-Right, Lower-Left corners).
        """
        tags = [
            ("ULLON", "ULLAT"),
            ("URLON", "URLAT"),
            ("LRLON", "LRLAT"),
            ("LLLON", "LLLAT"),
            ("ULLON", "ULLAT"),
        ]
        coords = []
        for lon_tag, lat_tag in tags:
            lon = get_xml_tag(xml, lon_tag)
            lat = get_xml_tag(xml, lat_tag)
            if lon and lat:
                coords.append(f"{lon} {lat}")
        geom_wkt = f"POLYGON(({', '.join(coords)}))"
        return geom_wkt

    def xml2poly(self, xml):
        """
        Convert XML corner coordinates to Shapely Polygon.

        Reads XML file and converts corner coordinates to a Shapely Polygon geometry.

        Parameters
        ----------
        xml : str
            Path to the XML file

        Returns
        -------
        shapely.geometry.Polygon
            Polygon geometry representing the image footprint
        """
        geom_wkt = self.xml2wkt(xml)
        return wkt.loads(geom_wkt)


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
            Sensor-agnostic scene dict (see module docstring).
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


# Registry of available sensor readers, in detection-priority order. The
# Pléiades reader detects strictly on the DIMAP root tag, while the WorldView
# reader matches any non-ortho XML, so Pléiades must be checked first.
SENSORS = [PleiadesMetadata, WorldViewMetadata]


def _common_base(paths):
    """Return a base directory for a list of files.

    Used to pick a working directory (for ``dg_mosaic`` outputs and pair
    naming) when a reader is built from an explicit file list rather than a
    directory. Returns the files' common parent directory, or the current
    working directory if they share no common parent or the list is empty.
    """
    paths = [os.path.abspath(p) for p in (paths or [])]
    if not paths:
        return os.getcwd()
    base = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
    # commonpath can return a file path if one entry is a prefix of another;
    # make sure we hand back a directory.
    return base if os.path.isdir(base) else os.path.dirname(base)


def resolve_xml_inputs(inputs, recursive=True):
    """Expand files, directories, and glob patterns into XML file paths.

    Lets a user point the tools at messy inputs without a fixed directory
    structure — e.g. ``geom_plot *.XML`` (already expanded by the shell),
    ``geom_plot scene1.xml scene2.xml``, ``geom_plot delivery_dir/``, or a mix.

    Each item of ``inputs`` may be:

    - a path to an XML file (included directly),
    - a directory (searched with :meth:`WorldViewMetadata._discover_xmls`, which
      is shallow-first and falls back to a recursive search), or
    - a glob pattern (expanded with :func:`glob.glob`).

    Results are de-duplicated (by absolute path) and returned sorted. Note that
    sensor-specific filtering of non-camera XMLs (``README.XML``, ortho
    products) is applied by the reader, not here.

    Parameters
    ----------
    inputs : str or os.PathLike or iterable of those
        One or more files, directories, and/or glob patterns.
    recursive : bool, optional
        Passed through to directory discovery and ``**`` glob expansion.
        Default True.

    Returns
    -------
    list of str
        Sorted, de-duplicated XML file paths.
    """
    if isinstance(inputs, (str, os.PathLike)):
        inputs = [inputs]

    collected = []
    for item in inputs:
        item = os.path.expanduser(str(item))
        if os.path.isdir(item):
            collected.extend(
                WorldViewMetadata._discover_xmls(item, recursive=recursive)
            )
        elif glob.has_magic(item):
            collected.extend(glob.glob(item, recursive=recursive))
        elif os.path.isfile(item):
            collected.append(item)
        else:
            logger.warning("Input does not exist, skipping: %s", item)

    seen = set()
    unique = []
    for path in collected:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return sorted(unique)


def sensor_for_inputs(inputs):
    """Detect and instantiate the appropriate sensor reader for explicit inputs.

    The file-list counterpart of :func:`sensor_for_directory`. Resolves
    ``inputs`` (files, directories, and/or globs) into a list of XML files,
    then returns an instance of the first registered reader whose
    :meth:`SensorMetadata.detect_files` matches.

    Parameters
    ----------
    inputs : str or os.PathLike or iterable of those
        One or more files, directories, and/or glob patterns.

    Returns
    -------
    SensorMetadata
        An initialized reader for the detected sensor.

    Raises
    ------
    ValueError
        If no XML files are found, or no registered sensor reader matches them.
    """
    image_list = resolve_xml_inputs(inputs)
    if not image_list:
        raise ValueError(
            "\n\nNo XML files found for the given input(s). "
            "Provide camera XML files, a directory, or a glob pattern.\n\n"
        )
    base = _common_base(image_list)
    for sensor_cls in SENSORS:
        if sensor_cls.detect_files(image_list):
            return sensor_cls(directory=base, image_list=image_list)
    raise ValueError(
        "\n\nNo supported sensor metadata files found among the given input(s). "
        f"Supported sensors: {', '.join(s.name for s in SENSORS)}.\n\n"
    )


def sensor_for_directory(directory):
    """Detect and instantiate the appropriate sensor reader for a directory.

    Iterates the ``SENSORS`` registry and returns an instance of the first
    reader whose :meth:`SensorMetadata.detect` matches the directory contents.

    Parameters
    ----------
    directory : str
        Path to directory containing camera/metadata files.

    Returns
    -------
    SensorMetadata
        An initialized reader for the detected sensor.

    Raises
    ------
    ValueError
        If no registered sensor reader matches the directory contents.
    """
    directory = os.path.expanduser(directory)
    # Two passes, mirroring the readers' shallow-first discovery: a sensor
    # whose metadata sits at the directory's top level wins over one that only
    # matches somewhere inside a nested delivery, regardless of registry order.
    for recursive in (False, True):
        for sensor_cls in SENSORS:
            if sensor_cls.detect(directory, recursive=recursive):
                return sensor_cls(directory)
    raise ValueError(
        "\n\nNo supported sensor metadata files found in directory. "
        f"Supported sensors: {', '.join(s.name for s in SENSORS)}.\n\n"
    )
