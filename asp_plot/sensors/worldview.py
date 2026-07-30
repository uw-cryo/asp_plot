"""Metadata reader for WorldView / DigitalGlobe-heritage XML camera files."""

import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import union_all, wkt

from asp_plot.sensors.base import (
    _NON_CAMERA_XML_RE,
    SensorMetadata,
    fill_scene_defaults,
    resolve_input_files,
)
from asp_plot.utils import get_xml_tag, run_subprocess_command

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def _tag_or_none(xml, tag):
    """Read an XML tag, returning None instead of raising when it is absent.

    Used for the optional scene-dict fields: ``dg_mosaic`` can strip image
    tags, and Multi (multispectral) products carry per-band TDI rather than a
    single ``TDILEVEL``, so a missing tag must degrade to "not provided"
    rather than crash the whole scene dict (issue #163; ASP wraps the same
    reads in try/catch in ``RPC_XML.cc``).
    """
    try:
        return get_xml_tag(xml, tag)
    except ValueError:
        return None


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
        # An explicit file list (e.g. shell-expanded ``stereo_geom *.XML``) is
        # used directly, minus non-camera XMLs; without one, the directory is
        # searched. Either way the files' common parent is the fallback base
        # directory for mosaic output and pair naming.
        self.directory, self.image_list = resolve_input_files(
            type(self), directory, image_list
        )

        if not self.image_list:
            raise ValueError(
                "\n\nMissing XML camera files. Cannot extract metadata without these.\n\n"
            )

    # The DG blocks a file must carry to be claimed as a WorldView camera XML.
    # ASP itself requires GEO/EPH/ATT/IMD to all parse before calling a file a
    # DG camera (``RPC_XML.cc`` read_xml); asp_plot reads IMD (summary tags),
    # EPH, and ATT, so those three are required here. ``dg_mosaic`` outputs
    # (``*.r100.xml``) retain all of them.
    _REQUIRED_DG_BLOCKS = frozenset(["IMD", "EPH", "ATT"])

    @classmethod
    def _is_camera_file(cls, path):
        """True if ``path`` is a DigitalGlobe-heritage camera XML.

        Content check (issue #162): the root element must be ``<isd>`` and the
        ``IMD``/``EPH``/``ATT`` blocks must be present. A name filter excludes
        ``README.XML``/``*ortho*.xml`` decoys before any parsing. Note the
        root tag alone is not sufficient — ASP's ``gen_aster`` camera XMLs
        also use an ``<isd>`` root but carry none of the DG blocks.

        Uses ``iterparse`` and returns as soon as all required blocks have
        been seen (they open near the top of the file, before the large
        ``GEO``/``RPB`` sections), so detection stays cheap during recursive
        delivery scans.
        """
        if _NON_CAMERA_XML_RE.search(os.path.basename(path)):
            return False
        try:
            remaining = set(cls._REQUIRED_DG_BLOCKS)
            root_seen = False
            for _, el in ET.iterparse(path, events=("start",)):
                if not root_seen:
                    if el.tag != "isd":
                        return False
                    root_seen = True
                elif el.tag in remaining:
                    remaining.discard(el.tag)
                    if not remaining:
                        return True
            return False
        except (ET.ParseError, OSError):
            return False

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

        The identity core (xml_fn, catid, sensor, date, geom) is read strictly — a
        camera XML without those is an error. The summary fields degrade to
        "not provided" (None/NaN) when their tags are absent (issue #163):
        ``dg_mosaic`` can strip image tags, and Multi products carry per-band
        TDI instead of a single ``TDILEVEL``.
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
                # Optional summary tags: a missing tag lands as None and drops
                # out of list_average, leaving NaN ("not provided").
                lst.append(_tag_or_none(xml, tag))
            else:
                # This returns a Shapely Polygon geometry
                lst.append(self.xml2poly(xml))

        tdi = _tag_or_none(xml, "TDILEVEL")
        d = {
            "xml_fn": xml,
            "catid": catid,
            "sensor": get_xml_tag(xml, "SATID"),
            "date": datetime.strptime(
                get_xml_tag(xml, "FIRSTLINETIME"), "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "scandir": _tag_or_none(xml, "SCANDIRECTION"),
            "tdi": int(tdi) if tdi is not None else None,
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

        return fill_scene_defaults(d)

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
