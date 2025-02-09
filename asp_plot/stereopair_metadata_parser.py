import logging
import os
from datetime import datetime, timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import osr
from shapely import union_all, wkt

from asp_plot.utils import get_xml_tag, glob_file

osr.UseExceptions()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# TODO: If this supports N scenes, should rename to SceneMetadataParser or something
class StereopairMetadataParser:
    def __init__(self, directory):
        self.directory = directory

    # TODO: N scenes
    # def get_scene_dict(self):
    #    return self.scene_dict()

    def get_pair_dict(self):
        catid_xmls = self.get_catid_xmls()

        id_dicts = []
        for catid, xml in catid_xmls.items():
            id_dicts.append(self.get_id_dict(catid, xml))

        id1_dict, id2_dict = id_dicts
        pairname = os.path.split(self.directory.rstrip("/\\"))[-1]
        return self.pair_dict(id1_dict, id2_dict, pairname)

    def get_catid_xmls(self):
        image_list = glob_file(self.directory, "*.[Xx][Mm][Ll]", all_files=True)
        if not image_list:
            raise ValueError(
                "\n\nMissing XML camera files in directory. Cannot extract metadata without these.\n\n"
            )

        # Get CATIDs
        catid_xmls = {}
        for xml_file in image_list:
            catid = get_xml_tag(xml_file, "CATID")
            catid_xmls[catid] = xml_file

        # TODO: need to improve logic and looping here and in get_id_dict for dictionary creation when
        # there are multiple XML files for a given scene

        return catid_xmls

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
            "id": catid,
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

    # Reads XML and returns a Shapely Polygon geometry
    def xml2poly(self, xml):
        geom_wkt = self.xml2wkt(xml)
        return wkt.loads(geom_wkt)

    def pair_dict(self, id1_dict, id2_dict, pairname):
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

        p = {}
        p["id1_dict"] = id1_dict
        p["id2_dict"] = id2_dict
        p["pairname"] = pairname

        self.get_pair_intersection(p)

        cdate = center_date([p["id1_dict"]["date"], p["id2_dict"]["date"]])
        p["cdate"] = cdate
        dt1 = p["id1_dict"]["date"]
        dt2 = p["id2_dict"]["date"]
        dt = abs(dt1 - dt2)
        p["dt"] = dt

        # TODO: migrate dgtools functions for BIE, asymmetry angles

        p["conv_ang"] = get_conv(
            p["id1_dict"]["meansataz"],
            p["id1_dict"]["meansatel"],
            p["id2_dict"]["meansataz"],
            p["id2_dict"]["meansatel"],
        )

        p["bh"] = get_bh(p["conv_ang"])
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
        # TODO: revist with GeoPanadas functions for overlay or cascading intersection
        def geom_intersection(geom_list):
            intsect = geom_list[0]
            valid = False
            for geom in geom_list[1:]:
                if intsect.intersects(geom):
                    valid = True
                    intsect = intsect.intersection(geom)
            if not valid:
                intsect = None
            return intsect

        def geom2local(geom, geom_crs="EPSG:4326"):
            local_proj = self.get_centroid_projection(geom, proj_type="ortho")
            gdf = gpd.GeoDataFrame(index=[0], crs=geom_crs, geometry=[geom])
            return gdf.to_crs(local_proj).geometry.squeeze()

        geom1 = p["id1_dict"]["geom"]
        geom2 = p["id2_dict"]["geom"]
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
