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

import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import union_all, wkt

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
    def detect(cls, directory):
        """Return True if this reader can handle the files in ``directory``.

        Parameters
        ----------
        directory : str
            Path to directory to inspect.

        Returns
        -------
        bool
            Whether this sensor's metadata files are present.
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

    Attributes
    ----------
    directory : str
        Path to directory containing XML files.
    image_list : list
        List of XML files found in the directory.
    """

    name = "WorldView"

    def __init__(self, directory):
        """
        Initialize the WorldView metadata reader.

        Parameters
        ----------
        directory : str
            Path to directory containing XML camera model files.

        Raises
        ------
        ValueError
            If no XML files are found in the directory.
        """
        super().__init__(directory)

        self.image_list = self._discover_xmls(self.directory)

        if not self.image_list:
            raise ValueError(
                "\n\nMissing XML camera files in directory. Cannot extract metadata without these.\n\n"
            )

    @staticmethod
    def _discover_xmls(directory):
        """Glob non-ortho XML camera files in ``directory``.

        Parameters
        ----------
        directory : str
            Path to directory to search.

        Returns
        -------
        list
            List of XML file paths, excluding ``*ortho*.xml`` files.
        """
        # glob_file returns None (not []) when nothing matches
        image_list = glob_file(directory, "*.[Xx][Mm][Ll]", all_files=True) or []
        # Drop potential *ortho*.xml files from image_list
        return [file for file in image_list if not re.search(r".*ortho.*\.xml", file)]

    @classmethod
    def detect(cls, directory):
        """Return True if non-ortho XML camera files are present.

        Parameters
        ----------
        directory : str
            Path to directory to inspect.

        Returns
        -------
        bool
            Whether WorldView XML camera files were found.
        """
        return bool(cls._discover_xmls(os.path.expanduser(directory)))

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

    def get_catid_xmls(self):
        """
        Get XML files associated with each catalog ID.

        Checks for multiple XML files for each catalog ID and handles mosaicking
        if needed.

        Returns
        -------
        dict
            Dictionary mapping catalog IDs to XML file paths

        Notes
        -----
        If more than two XML files are found, they will be mosaicked using
        dg_mosaic before proceeding.
        """
        # First check for multiple XML files and dg_mosaic if needed
        if len(self.image_list) > 2:
            print(
                "\nMore than two XML files found in directory. Mosaicking before proceeding.\n"
            )
            self.mosaic_multiple_xmls()

        # Get CATIDs
        catid_xmls = {}
        for xml_file in self.image_list:
            catid = get_xml_tag(xml_file, "CATID")
            catid_xmls[catid] = xml_file

        # TODO: need to improve logic and looping here and in get_id_dict for dictionary creation when
        # there are multiple XML files for a given scene
        # use ~/Desktop/asp-plot-examples/antarctica/tiled_xmls_example for testing this

        return catid_xmls

    def mosaic_multiple_xmls(self):
        """
        Mosaic multiple XML files for each catalog ID.

        Uses dg_mosaic to merge multiple XML files for the same catalog ID
        into a single XML file. This is needed when a scene is composed of
        multiple image tiles.

        Returns
        -------
        None
            Updates the image_list attribute with mosaicked XML files

        Notes
        -----
        Requires dg_mosaic from the NASA Ames Stereo Pipeline to be installed
        and available in the system path.
        """
        # Drop existing *.r100.* and *.r50.* files from image_list if they are present
        self.image_list = [
            file
            for file in self.image_list
            if not re.search(r"\.r100\..*|\.r50\..*", file)
        ]

        # Group XML files by CATID
        catid_xml_dict = {}
        for xml_file in self.image_list:
            catid = get_xml_tag(xml_file, "CATID")
            if catid not in catid_xml_dict:
                catid_xml_dict[catid] = []
            catid_xml_dict[catid].append(xml_file)

        # Convert lists to space-separated strings
        catid_xml_dict = {
            catid: " ".join(xml_files) for catid, xml_files in catid_xml_dict.items()
        }

        # Run dg_mosaic with: dg_mosaic --skip-tif-gen --output-prefix <NAME> <SPACE SEPARATED XML FILES>
        output_xmls = []
        for catid, xml_files in catid_xml_dict.items():
            output_xml = os.path.join(self.directory, f"{catid}_asp_plot_dg_mosaic")
            output_xml_r100 = f"{output_xml}.r100.xml"

            if not os.path.exists(output_xml_r100):
                # Build the command string instead of a list, needed for subprocess call, .split() below
                command = (
                    f"dg_mosaic --skip-tif-gen --output-prefix {output_xml} {xml_files}"
                )

                print(f"\nRunning dg_mosaic with command: {command}\n")

                # Run the command
                run_subprocess_command(command.split())
            else:
                print(f"\nUsing existing mosaicked XML file: {output_xml_r100}\n")

            output_xmls.append(output_xml_r100)

        # Then create the new image list with just the mosaicked XML files
        self.image_list = []
        for output_xml in output_xmls:
            self.image_list.append(output_xml)

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


# Registry of available sensor readers, in detection-priority order.
SENSORS = [WorldViewMetadata]


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
    for sensor_cls in SENSORS:
        if sensor_cls.detect(directory):
            return sensor_cls(directory)
    raise ValueError(
        "\n\nNo supported sensor metadata files found in directory. "
        f"Supported sensors: {', '.join(s.name for s in SENSORS)}.\n\n"
    )
