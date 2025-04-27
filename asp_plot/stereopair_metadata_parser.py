import logging
import os
import re
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import osr
from shapely import union_all, wkt

from asp_plot.utils import get_xml_tag, glob_file, run_subprocess_command

osr.UseExceptions()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# TODO: When this supports N scenes, should rename to StereoMetadataParser
class StereopairMetadataParser:
    def __init__(self, directory):
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
        catid_dicts = self.get_catid_dicts()
        catid1_dict, catid2_dict = catid_dicts
        pairname = os.path.split(self.directory.rstrip("/\\"))[-1]
        return self.pair_dict(catid1_dict, catid2_dict, pairname)

    def get_catid_dicts(self):
        catid_xmls = self.get_catid_xmls()
        catid_dicts = []
        for catid, xml in catid_xmls.items():
            catid_dicts.append(self.get_id_dict(catid, xml))
        return catid_dicts

    def get_catid_xmls(self):
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
        def list_average(list):
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
        e = get_xml_tag(xml, "EPHEMLIST", all=True)
        # Could get fancy with structured array here
        # point_num, Xpos, Ypos, Zpos, Xvel, Yvel, Zvel, covariance matrix (6 elements)
        # dtype=[('point', 'i4'), ('Xpos', 'f8'), ('Ypos', 'f8'), ('Zpos', 'f8'), ('Xvel', 'f8') ...]
        # All coordinates are ECF, meters, meters/sec, m^2
        return np.array([i.split() for i in e], dtype=np.float64)

    def getEphem_gdf(self, xml):
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
        """Reads XML and returns a Shapely Polygon geometry"""
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
        """Get local projection centered on geometry centroid

        Args:
            geom: Shapely geometry object
            proj_type: Type of projection ('tmerc' or 'ortho')

        Returns:
            str: Proj4 string for local projection
        """
        centroid = geom.centroid
        return f"+proj={proj_type} +lat_0={centroid.y:0.7f} +lon_0={centroid.x:0.7f}"

    def get_pair_intersection(self, p):
        def geom_intersection(geom_list):
            gdfs = [
                gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326") for geom in geom_list
            ]
            result = gdfs[0]
            for gdf in gdfs[1:]:
                result = gpd.overlay(result, gdf, how="intersection")
            return result.geometry.iloc[0] if not result.empty else None

        def geom2local(geom, geom_crs="EPSG:4326"):
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
