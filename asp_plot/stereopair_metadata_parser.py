import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime
from osgeo import ogr, osr, gdal


class StereopairMetadataParser:
    def __init__(self, stereopair_directory):
        self.stereopair_directory = stereopair_directory

    def get_pair_dict(self):
        ids = self.get_ids()
        id1_dict = self.get_id_dict(ids[0])
        id2_dict = self.get_id_dict(ids[1])
        pairname = os.path.split(self.stereopair_directory)[-1]
        return self.pair_dict(id1_dict, id2_dict, pairname)

    def get_ids(self):

        def get_id(filename):
            import re

            ids = re.findall("10[123456][0-9a-fA-F]+00", filename)
            return list(set(ids))

        image_list = glob.glob(
            os.path.join(self.stereopair_directory, "*.[Xx][Mm][Ll]")
        )
        ids = [get_id(f) for f in image_list]
        ids = sorted(set(item for sublist in ids if sublist for item in sublist))
        return ids

    def get_id_dict(self, id):

        def list_average(list):
            return np.round(pd.Series(list, dtype=float).dropna().mean(), 2)

        def geom_union(geom_list):
            union = geom_list[0]
            for geom in geom_list[1:]:
                union = union.Union(geom)
            return union

        xml_list = glob.glob(
            os.path.join(self.stereopair_directory, f"*{id:}*.[Xx][Mm][Ll]")
        )

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
            "ground_geom": [],
        }

        for xml in xml_list:
            for tag, lst in attributes.items():
                if tag != "ground_geom":
                    lst.append(self.getTag(xml, tag))
                else:
                    lst.append(self.xml2geom(xml))

        d = {
            "id": str(id),
            "sensor": self.getTag(xml_list[0], "SATID"),
            "date": datetime.strptime(
                self.getTag(xml_list[0], "FIRSTLINETIME"), "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "scandir": self.getTag(xml_list[0], "SCANDIRECTION"),
            "tdi": int(self.getTag(xml_list[0], "TDILEVEL")),
            "ground_geom": geom_union(attributes["ground_geom"]),
        }

        for tag, lst in attributes.items():
            if tag != "ground_geom":
                d[tag.lower()] = list_average(lst)

        return d

    def getTag(self, xml, tag):
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml)
        elem = tree.find(".//%s" % tag)
        if elem is not None:
            return elem.text

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
            lon = self.getTag(xml, lon_tag)
            lat = self.getTag(xml, lat_tag)
            if lon and lat:  # Ensure both longitude and latitude are found
                coords.append(f"{lon} {lat}")
        geom_wkt = f"POLYGON(({', '.join(coords)}))"
        return geom_wkt

    def xml2geom(self, xml):
        geom_wkt = self.xml2wkt(xml)
        geom = ogr.CreateGeometryFromWkt(geom_wkt)
        # Define WGS84 srs; mpd = 111319.9
        wgs_srs = osr.SpatialReference()
        wgs_srs.SetWellKnownGeogCS("WGS84")
        # Hack for GDAL3, should reorder with (lat,lon) as specified
        if int(gdal.__version__.split(".")[0]) >= 3:
            wgs_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geom.AssignSpatialReference(wgs_srs)
        return geom
