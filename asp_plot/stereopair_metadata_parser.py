import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from osgeo import ogr, osr, gdal
from shapely import wkt
from asp_plot.utils import get_xml_tag


class StereopairMetadataParser:
    def __init__(self, directory):
        self.directory = directory

    def get_pair_dict(self):
        ids = self.get_ids()
        id1_dict = self.get_id_dict(ids[0])
        id2_dict = self.get_id_dict(ids[1])
        pairname = os.path.split(self.directory.rstrip("/\\"))[-1]
        return self.pair_dict(id1_dict, id2_dict, pairname)

    def get_ids(self):

        def get_id(filename):
            import re

            ids = re.findall("10[123456][0-9a-fA-F]+00", filename)
            return list(set(ids))

        image_list = glob.glob(os.path.join(self.directory, "*.[Xx][Mm][Ll]"))
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

        xml_list = glob.glob(os.path.join(self.directory, f"*{id:}*.[Xx][Mm][Ll]"))

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

        for xml in xml_list:
            for tag, lst in attributes.items():
                if tag != "geom":
                    lst.append(get_xml_tag(xml, tag))
                else:
                    lst.append(self.xml2geom(xml))

        d = {
            "id": str(id),
            "sensor": get_xml_tag(xml_list[0], "SATID"),
            "date": datetime.strptime(
                get_xml_tag(xml_list[0], "FIRSTLINETIME"), "%Y-%m-%dT%H:%M:%S.%fZ"
            ),
            "scandir": get_xml_tag(xml_list[0], "SCANDIRECTION"),
            "tdi": int(get_xml_tag(xml_list[0], "TDILEVEL")),
            "geom": geom_union(attributes["geom"]),
        }

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

    def xml2geom(self, xml):
        geom_wkt = self.xml2wkt(xml)
        geom = ogr.CreateGeometryFromWkt(geom_wkt)
        wgs_srs = self.get_wgs_srs()
        geom.AssignSpatialReference(wgs_srs)
        return geom

    def xml2gdf(self, xml, init_crs="EPSG:4326"):
        poly = self.xml2poly(xml)
        gdf = gpd.GeoDataFrame(
            {"idx": [0], "geometry": poly}, geometry="geometry", crs=init_crs
        )
        return gdf

    def xml2poly(self, xml):
        geom_wkt = self.xml2wkt(xml)
        return wkt.loads(geom_wkt)

    def get_wgs_srs(self):
        # Define WGS84 srs
        # mpd = 111319.9
        wgs_srs = osr.SpatialReference()
        wgs_srs.SetWellKnownGeogCS("WGS84")
        # GDAL3 hack
        if int(gdal.__version__.split(".")[0]) >= 3:
            wgs_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        return wgs_srs

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

        p["conv_ang"] = get_conv(
            p["id1_dict"]["meansataz"],
            p["id1_dict"]["meansatel"],
            p["id2_dict"]["meansataz"],
            p["id2_dict"]["meansatel"],
        )

        p["bh"] = get_bh(p["conv_ang"])
        return p

    def get_pair_intersection(self, p):
        def geom_intersection(geom_list):
            intsect = geom_list[0]
            valid = False
            for geom in geom_list[1:]:
                if intsect.Intersects(geom):
                    valid = True
                    intsect = intsect.Intersection(geom)
            if not valid:
                intsect = None
            return intsect

        def geom2localortho(geom):
            wgs_srs = self.get_wgs_srs()
            cx, cy = geom.Centroid().GetPoint_2D()
            lon, lat, z = self.coordinate_transformation_helper(
                cx, cy, 0, geom.GetSpatialReference(), wgs_srs
            )
            local_srs = osr.SpatialReference()
            local_proj = f"+proj=ortho +lat_0={lat:0.7f} +lon_0={lon:0.7f} +datum=WGS84 +units=m +no_defs "
            local_srs.ImportFromProj4(local_proj)
            local_geom = geom_dup(geom)
            geom_transform(local_geom, local_srs)
            return local_geom

        def geom_dup(geom):
            g = ogr.CreateGeometryFromWkt(geom.ExportToWkt())
            g.AssignSpatialReference(geom.GetSpatialReference())
            return g

        def geom_transform(geom, t_srs):
            s_srs = geom.GetSpatialReference()
            if not s_srs.IsSame(t_srs):
                ct = osr.CoordinateTransformation(s_srs, t_srs)
                geom.Transform(ct)
                geom.AssignSpatialReference(t_srs)

        geom1 = p["id1_dict"]["geom"]
        geom2 = p["id2_dict"]["geom"]
        intersection = geom_intersection([geom1, geom2])
        p["intersection"] = intersection
        p["intersection_poly"] = wkt.loads(intersection.ExportToWkt())
        intersection_local = geom2localortho(intersection)
        local_srs = intersection_local.GetSpatialReference()
        # This recomputes for local orthographic - important for width/height calculations
        geom1_local = geom_dup(geom1)
        geom_transform(geom1_local, local_srs)
        geom2_local = geom_dup(geom2)
        geom_transform(geom2_local, local_srs)
        if intersection is not None:
            # Area calc shouldn't matter too much
            intersection_area = intersection_local.GetArea()
            p["intersection_area"] = np.round(intersection_area / 1e6, 2)
            perc = (
                100.0 * intersection_area / geom1_local.GetArea(),
                100 * intersection_area / geom2_local.GetArea(),
            )
            perc = (np.round(perc[0], 2), np.round(perc[1], 2))
            p["intersection_area_perc"] = perc
        else:
            p["intersection_area"] = None
            p["intersection_area_perc"] = None

    def coordinate_transformation_helper(self, x, y, z, in_srs, out_srs):
        def common_mask(ma_list, apply=False):
            a = np.ma.array(ma_list, shrink=False)
            mask = np.ma.getmaskarray(a).any(axis=0)
            if apply:
                return [np.ma.array(b, mask=mask) for b in ma_list]
            else:
                return mask

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        if x.shape != y.shape:
            sys.exit("Inconsistent number of x and y points")
        valid_idx = None
        # Handle case where we have x array, y array, but a constant z (e.g., 0.0)
        if z.shape != x.shape:
            # If a constant elevation is provided
            if z.shape[0] == 1:
                orig_z = z
                z = np.zeros_like(x)
                z[:] = orig_z
                if np.ma.is_masked(x):
                    z[np.ma.getmaskarray(x)] = np.ma.masked
            else:
                sys.exit("Inconsistent number of z and x/y points")
        # If any of the inputs is masked, only transform points with all three coordinates available
        if np.ma.is_masked(x) or np.ma.is_masked(y) or np.ma.is_masked(z):
            x = np.ma.array(x)
            y = np.ma.array(y)
            z = np.ma.array(z)
            valid_idx = ~(common_mask([x, y, z]))
            # Prepare (x,y,z) tuples
            xyz = np.array([x[valid_idx], y[valid_idx], z[valid_idx]]).T
        else:
            xyz = np.array([x.ravel(), y.ravel(), z.ravel()]).T
        # Define coordinate transformation
        coordinate_transformation = osr.CoordinateTransformation(in_srs, out_srs)
        # Loop through each point
        xyz2 = np.array(
            [
                coordinate_transformation.TransformPoint(xi, yi, zi)
                for (xi, yi, zi) in xyz
            ]
        ).T
        # If single input coordinate
        if xyz2.shape[1] == 1:
            xyz2 = xyz2.squeeze()
            x2, y2, z2 = xyz2[0], xyz2[1], xyz2[2]
        else:
            # Fill in masked array
            if valid_idx is not None:
                x2 = np.zeros_like(x)
                y2 = np.zeros_like(y)
                z2 = np.zeros_like(z)
                x2[valid_idx] = xyz2[0]
                y2[valid_idx] = xyz2[1]
                z2[valid_idx] = xyz2[2]
            else:
                x2 = xyz2[0].reshape(x.shape)
                y2 = xyz2[1].reshape(y.shape)
                z2 = xyz2[2].reshape(z.shape)
        return x2, y2, z2
