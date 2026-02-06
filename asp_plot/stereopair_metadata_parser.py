import logging
import os
import re
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import osr
from shapely import union_all, wkt

from asp_plot.utils import get_utm_epsg, get_xml_tag, glob_file, run_subprocess_command

osr.UseExceptions()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# TODO: When this supports N scenes, should rename to StereoMetadataParser
class StereopairMetadataParser:
    """
    Parse metadata from satellite sensor XML files for stereo pairs.

    This class parses Digital Globe/Maxar or similar satellite XML files to extract
    metadata and compute stereo geometry parameters. It handles single XML files
    as well as multiple XML files per scene, where mosaicking is required.

    Attributes
    ----------
    directory : str
        Path to directory containing XML files
    image_list : list
        List of XML files found in the directory

    Examples
    --------
    >>> parser = StereopairMetadataParser('/path/to/stereo/directory')
    >>> pair_dict = parser.get_pair_dict()
    >>> print(f"Convergence angle: {pair_dict['conv_ang']}")
    >>> print(f"Base-to-height ratio: {pair_dict['bh']}")
    """

    def __init__(self, directory):
        """
        Initialize the StereopairMetadataParser.

        Parameters
        ----------
        directory : str
            Path to directory containing XML camera model files

        Raises
        ------
        ValueError
            If no XML files are found in the directory
        """
        self.directory = directory

        self.image_list = glob_file(self.directory, "*.[Xx][Mm][Ll]", all_files=True)

        # Drop potential *ortho*.xml files from image_list
        self.image_list = [
            file for file in self.image_list if not re.search(r".*ortho.*\.xml", file)
        ]

        if not self.image_list:
            raise ValueError(
                "\n\nMissing XML camera files in directory. Cannot extract metadata without these.\n\n"
            )

    # TODO: This method assumes that only two scenes are captured with get_catid_dicts
    # Should be updated to support more than two scenes, or need a separate method for N scenes
    def get_pair_dict(self):
        """
        Get a dictionary with all stereo pair information.

        Creates a comprehensive dictionary containing stereo pair information,
        including convergence angle, base-to-height ratio, bisector elevation angle,
        asymmetry angle, and more.

        Returns
        -------
        dict
            Dictionary with stereo pair information and geometry parameters

        Notes
        -----
        Currently only supports exactly two scenes. Future versions will support N scenes.
        """
        catid_dicts = self.get_catid_dicts()
        catid1_dict, catid2_dict = catid_dicts
        pairname = os.path.split(self.directory.rstrip("/\\"))[-1]
        return self.pair_dict(catid1_dict, catid2_dict, pairname)

    def get_catid_dicts(self):
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
        # use ~/Dropbox/UW_Shean/WV/antarctica/tiled_xmls_example for testing this

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

        # Add Ephemeris GeoDataFrame and Footprint GeoDataFrame
        if geteph:
            d["eph_gdf"] = self.getEphem_gdf(xml)
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
        names.extend(["{}_cov".format(n) for n in names[1:7]])
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

    def pair_dict(self, catid1_dict, catid2_dict, pairname):
        def center_date(dt_list):
            dt_list_sort = sorted(dt_list)
            dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
            avg_timedelta = sum(dt_list_sort_rel, timedelta()) / len(dt_list_sort_rel)
            return dt_list_sort[0] + avg_timedelta

        def get_conv(az1, el1, az2, el2):
            conv_ang = np.rad2deg(
                np.arccos(
                    np.sin(np.deg2rad(el1)) * np.sin(np.deg2rad(el2))
                    + np.cos(np.deg2rad(el1))
                    * np.cos(np.deg2rad(el2))
                    * np.cos(np.deg2rad(az1 - az2))
                )
            )
            return np.round(conv_ang, 2)

        def get_bh(conv_ang):
            bh = 2 * np.tan(np.deg2rad(conv_ang / 2.0))
            return np.round(bh, 2)

        def get_bie(az1, el1, az2, el2):
            """Calculate Bisector Elevation Angle for stereo pair

            From Jeong and Kim 2014: https://www.ingentaconnect.com/content/asprs/pers/2014/00000080/00000007/art00004?crawler=true

            Parameters
            ------------
            el1: numeric
                satellite elevation angle during acquisition of first image
            az1: numeric
                satellite azimuth angle during acquisition of first image
            el2: numeric
                satellite elevation angle during acquisition of second image
            az2: numeric
                satellite azimuth angle during acquisition of second image

            Returns
            ------------
            bie: numeric
                Bisector Elevation Angle for input stereo pair
            """
            num = np.sin(np.deg2rad(el1)) + np.sin(np.deg2rad(el2))
            denom = np.sqrt(2) * np.sqrt(
                1
                + np.cos(np.deg2rad(az1 - az2))
                * np.cos(np.deg2rad(el1))
                * np.cos(np.deg2rad(el2))
                + np.sin(np.deg2rad(el1)) * np.sin(np.deg2rad(el2))
            )
            bie = np.rad2deg(np.arcsin(num / denom))
            return np.round(bie, 2)

        def get_asymmetry_angle(sat1_pos, sat2_pos, ground_point):
            """Calculate asymmetry angle between satellite positions and ground point

            Parameters
            ------------
            sat1_pos: np.array
                3-D position of satellite during acquisition of first image (in ECEF)
            sat2_pos: np.array
                3-D position of satellite during acquisition of second image (in ECEF)
            ground_point: np.array
                3-D position of ground point viewed by both satellites (in ECEF)

            Returns
            ------------
            asymmetry_angle: numeric
                asymmetry_angle for the stereo pair in degrees
            """
            R = ground_point  # radius vector for ground point
            R01 = sat1_pos  # radius vector for satellite position at time t1
            R02 = sat2_pos  # radius vector for satellite position at time t2
            L1 = R - R01  # first pointing vector
            L2 = R - R02  # second pointing vector
            q1 = -L1 / np.linalg.norm(L1)  # first pointing (unit) vector
            q2 = -L2 / np.linalg.norm(L2)  # second pointing (unit) vector
            Zt = R / np.linalg.norm(
                R
            )  # geocentric radius vector for ground point (from origin to up)

            # calculate projection of geocentric vector radius vector on the convergence plane (contd. on next line)
            # convergence plane is formed by the two pointing vectors and the baseline vector
            A = np.cross(q1.tolist(), q2.tolist()) / np.linalg.norm(
                np.cross(q1.tolist(), q2.tolist())
            )
            num = np.cross(A, np.cross(Zt, A))
            denom = np.linalg.norm(num)
            Zt_si = num / denom

            # calculate bisector for convergence angles
            B = (q1 + q2) / np.linalg.norm((q1 + q2))

            # find angle between bisector angle and projection of geocentric ground point radius vector on the convergence plane
            asymmetry_angle = np.rad2deg(np.arccos(np.dot(B, Zt_si)))
            return np.round(asymmetry_angle, 2)

        # Create the pair dictionary and fill it in
        p = {}
        p["catid1_dict"] = catid1_dict
        p["catid2_dict"] = catid2_dict
        p["pairname"] = pairname

        self.get_pair_intersection(p)

        cdate = center_date([p["catid1_dict"]["date"], p["catid2_dict"]["date"]])
        p["cdate"] = cdate
        dt1 = p["catid1_dict"]["date"]
        dt2 = p["catid2_dict"]["date"]
        dt = abs(dt1 - dt2)
        p["dt"] = dt

        p["conv_ang"] = get_conv(
            p["catid1_dict"]["meansataz"],
            p["catid1_dict"]["meansatel"],
            p["catid2_dict"]["meansataz"],
            p["catid2_dict"]["meansatel"],
        )

        p["bh"] = get_bh(p["conv_ang"])

        p["bie"] = get_bie(
            p["catid1_dict"]["meansataz"],
            p["catid1_dict"]["meansatel"],
            p["catid2_dict"]["meansataz"],
            p["catid2_dict"]["meansatel"],
        )

        if "eph_gdf" in p["catid1_dict"] and "eph_gdf" in p["catid2_dict"]:
            sat1_pos = (
                p["catid1_dict"]["eph_gdf"]
                .iloc[len(p["catid1_dict"]["eph_gdf"]) // 2][["x", "y", "z"]]
                .values
            )
            sat2_pos = (
                p["catid2_dict"]["eph_gdf"]
                .iloc[len(p["catid2_dict"]["eph_gdf"]) // 2][["x", "y", "z"]]
                .values
            )

            # Use intersection centroid as ground point
            if p["intersection"] is not None:
                ground_point = (
                    gpd.GeoDataFrame(
                        geometry=[p["intersection"].centroid], crs="EPSG:4326"
                    )
                    .to_crs("EPSG:4978")
                    .geometry.values[0]
                    .coords[0]
                )

                # We set the z-coordinate to 0.0, instead of relying on DEM search with internet connection
                ground_point = np.array([ground_point[0], ground_point[1], 0.0])

                p["asymmetry_angle"] = get_asymmetry_angle(
                    sat1_pos, sat2_pos, ground_point
                )

        return p

    def get_centroid_projection(self, geom, proj_type="tmerc"):
        """
        Get a local projection centered on geometry centroid.

        Creates a custom projection string centered on the centroid of the input
        geometry, which minimizes distortion for local analyses.

        Parameters
        ----------
        geom : shapely.geometry.BaseGeometry
            Shapely geometry object whose centroid will be used as projection center
        proj_type : str, optional
            Type of projection to use, default is "tmerc" (Transverse Mercator)
            Other options include "ortho" (Orthographic)

        Returns
        -------
        str
            Proj4 string for local projection centered on geometry centroid

        Examples
        --------
        >>> parser = StereopairMetadataParser('/path/to/stereo/directory')
        >>> pair_dict = parser.get_pair_dict()
        >>> local_proj = parser.get_centroid_projection(pair_dict['intersection'])
        >>> print(local_proj)
        +proj=tmerc +lat_0=XX.XXXXXXX +lon_0=XX.XXXXXXX
        """
        centroid = geom.centroid
        return f"+proj={proj_type} +lat_0={centroid.y:0.7f} +lon_0={centroid.x:0.7f}"

    def get_pair_intersection(self, p):
        """
        Calculate intersection geometry and area for a stereo pair.

        Computes the intersection between two image footprints, calculates its area,
        and the percentage of each image covered by the intersection.

        Parameters
        ----------
        p : dict
            Stereo pair dictionary to update with intersection information

        Returns
        -------
        None
            Updates the input dictionary with intersection geometry and area information

        Notes
        -----
        The dictionary 'p' is updated with the following keys:
        - intersection: Shapely geometry representing the intersection
        - intersection_area: Area in square kilometers
        - intersection_area_perc: Tuple with percentages of each image covered by the intersection

        Areas are calculated in a local orthographic projection to minimize distortion.
        """

        def geom_intersection(geom_list):
            """Find the intersection of multiple geometries."""
            gdfs = [
                gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326") for geom in geom_list
            ]
            result = gdfs[0]
            for gdf in gdfs[1:]:
                result = gpd.overlay(result, gdf, how="intersection")
            return result.geometry.iloc[0] if not result.empty else None

        def geom2local(geom, geom_crs="EPSG:4326"):
            """Convert geometry to a local projection for accurate area calculation."""
            local_proj = self.get_centroid_projection(geom, proj_type="ortho")
            gdf = gpd.GeoDataFrame(index=[0], crs=geom_crs, geometry=[geom])
            return gdf.to_crs(local_proj).geometry.squeeze()

        geom1 = p["catid1_dict"]["geom"]
        geom2 = p["catid2_dict"]["geom"]
        intersection = geom_intersection([geom1, geom2])
        p["intersection"] = intersection
        intersection_local = geom2local(intersection)
        if intersection is not None:
            # Area calc shouldn't matter too much
            intersection_area = intersection_local.area
            p["intersection_area"] = np.round(intersection_area / 1e6, 2)
            perc = (
                100.0 * intersection_area / geom2local(geom1).area,
                100 * intersection_area / geom2local(geom2).area,
            )
            perc = (np.round(perc[0], 2), np.round(perc[1], 2))
            p["intersection_area_perc"] = perc
        else:
            p["intersection_area"] = None
            p["intersection_area_perc"] = None

    def get_pair_utm_epsg(self):
        """
        Get the UTM EPSG code for the stereo pair's intersection area.

        Uses the centroid of the pair intersection footprint to determine
        the appropriate UTM zone.

        Returns
        -------
        int
            UTM EPSG code (e.g., 32616 for UTM Zone 16N)
        """
        pair = self.get_pair_dict()
        centroid = pair["intersection"].centroid
        return get_utm_epsg(centroid.x, centroid.y)

    def get_intersection_bounds(self, epsg=None):
        """
        Get the bounding box of the stereo pair intersection area.

        Returns the intersection of both image footprints as a bounding box,
        optionally reprojected to a given CRS.

        Parameters
        ----------
        epsg : int, optional
            EPSG code to reproject bounds into (e.g., 32616 for UTM Zone 16N).
            If None, returns bounds in EPSG:4326 (longitude/latitude).

        Returns
        -------
        tuple
            (min_x, min_y, max_x, max_y) in the requested CRS
        """
        pair = self.get_pair_dict()
        intersection = pair["intersection"]
        if epsg is not None:
            intersection = (
                gpd.GeoDataFrame(geometry=[intersection], crs="EPSG:4326")
                .to_crs(f"EPSG:{epsg}")
                .geometry.iloc[0]
            )
        return intersection.bounds

    def get_scene_bounds(self):
        """
        Get the geographic bounds of the union of all scene footprints.

        Computes the union of both image footprints and returns the
        bounding box in longitude/latitude (EPSG:4326).

        Returns
        -------
        tuple
            (min_lon, min_lat, max_lon, max_lat)
        """
        pair = self.get_pair_dict()
        scene_union = union_all(
            [pair["catid1_dict"]["geom"], pair["catid2_dict"]["geom"]]
        )
        return scene_union.bounds
