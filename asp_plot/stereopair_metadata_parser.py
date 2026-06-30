import logging
import os
from datetime import timedelta
from itertools import combinations

import geopandas as gpd
import numpy as np
from osgeo import osr
from shapely import union_all

from asp_plot.sensors import sensor_for_directory, sensor_for_inputs
from asp_plot.utils import get_utm_epsg

osr.UseExceptions()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_convergence_angle(az1, el1, az2, el2):
    """Calculate convergence angle between two satellite viewing directions.

    Uses the spherical law of cosines to compute the angle between two
    unit vectors defined by their azimuth and elevation angles.

    Parameters
    ----------
    az1 : numeric
        Satellite azimuth angle for first image (degrees)
    el1 : numeric
        Satellite elevation angle for first image (degrees)
    az2 : numeric
        Satellite azimuth angle for second image (degrees)
    el2 : numeric
        Satellite elevation angle for second image (degrees)

    Returns
    -------
    float
        Convergence angle in degrees, rounded to 2 decimal places

    References
    ----------
    Jeong & Kim (2016), PE&RS 82(8), 625-633, Eq. 1
    """
    conv_ang = np.rad2deg(
        np.arccos(
            np.sin(np.deg2rad(el1)) * np.sin(np.deg2rad(el2))
            + np.cos(np.deg2rad(el1))
            * np.cos(np.deg2rad(el2))
            * np.cos(np.deg2rad(az1 - az2))
        )
    )
    return np.round(conv_ang, 2)


def get_bh_ratio(conv_ang):
    """Calculate base-to-height ratio from convergence angle.

    Parameters
    ----------
    conv_ang : numeric
        Convergence angle in degrees

    Returns
    -------
    float
        Base-to-height ratio, rounded to 2 decimal places
    """
    bh = 2 * np.tan(np.deg2rad(conv_ang / 2.0))
    return np.round(bh, 2)


def get_bie_angle(az1, el1, az2, el2):
    """Calculate Bisector Elevation Angle for a stereo pair.

    The BIE is the elevation angle of the bisector of the two viewing
    directions. Higher BIE means less oblique epipolar geometry and
    better positioning accuracy.

    Parameters
    ----------
    az1 : numeric
        Satellite azimuth angle for first image (degrees)
    el1 : numeric
        Satellite elevation angle for first image (degrees)
    az2 : numeric
        Satellite azimuth angle for second image (degrees)
    el2 : numeric
        Satellite elevation angle for second image (degrees)

    Returns
    -------
    float
        Bisector Elevation Angle in degrees, rounded to 2 decimal places

    References
    ----------
    Jeong & Kim (2014), PE&RS 80(7), 653-662
    Jeong & Kim (2016), PE&RS 82(8), 625-633, Eq. 2
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
    """Calculate asymmetry angle between satellite positions and ground point.

    The asymmetry angle measures how far the bisector of the two viewing
    rays deviates from the local vertical, projected onto the convergence
    plane. An asymmetry of 0 means perfectly symmetric stereo geometry.

    Parameters
    ----------
    sat1_pos : numpy.ndarray
        3-D position of satellite during acquisition of first image (in ECEF)
    sat2_pos : numpy.ndarray
        3-D position of satellite during acquisition of second image (in ECEF)
    ground_point : numpy.ndarray
        3-D position of ground point viewed by both satellites (in ECEF)

    Returns
    -------
    float
        Asymmetry angle in degrees, rounded to 2 decimal places

    References
    ----------
    Jeong & Kim (2014), PE&RS 80(7), 653-662
    Jeong & Kim (2016), PE&RS 82(8), 625-633, Eq. 3
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


# TODO: When this supports N scenes, should rename to StereoMetadataParser
class StereopairMetadataParser:
    """
    Parse metadata for a stereo pair and compute stereo-geometry parameters.

    This class is sensor-agnostic: the work of discovering scene files and
    extracting per-scene metadata is delegated to a sensor-specific reader (see
    :mod:`asp_plot.sensors`), chosen automatically by inspecting the directory
    contents. The parser then computes pair-level geometry (convergence angle,
    base-to-height ratio, bisector elevation angle, asymmetry angle, footprint
    intersection, bounds) from the resulting scene dictionaries.

    Adding support for a new sensor (ASTER, HiRISE, etc.) is a matter of writing
    a new :class:`asp_plot.sensors.SensorMetadata` subclass; no changes to this
    class are required.

    Attributes
    ----------
    directory : str
        Path to directory containing the scene metadata files
    reader : asp_plot.sensors.SensorMetadata
        The detected sensor-specific metadata reader
    image_list : list
        List of scene metadata files found in the directory (delegated to the reader)

    Examples
    --------
    >>> parser = StereopairMetadataParser('/path/to/stereo/directory')
    >>> pair_dict = parser.get_pair_dict()
    >>> print(f"Convergence angle: {pair_dict['conv_ang']}")
    >>> print(f"Base-to-height ratio: {pair_dict['bh']}")
    """

    def __init__(self, directory=None, inputs=None):
        """
        Initialize the StereopairMetadataParser.

        Detects the sensor and builds the matching metadata reader, either from
        a ``directory`` or from an explicit list of ``inputs`` (files,
        directories, and/or glob patterns, e.g. a ``geom_plot *.XML`` call).
        Exactly one of the two should be given.

        Parameters
        ----------
        directory : str, optional
            Path to directory containing camera model / metadata files.
        inputs : str or os.PathLike or iterable of those, optional
            Explicit files, directories, and/or glob patterns to use instead of
            a single ``directory``. Takes precedence when both are given.

        Raises
        ------
        ValueError
            If neither ``directory`` nor ``inputs`` is given, or if no supported
            sensor metadata files are found.
        """
        if inputs is None and directory is None:
            raise ValueError("Provide either a directory or inputs.")

        if inputs is not None:
            self.reader = sensor_for_inputs(inputs)
            self.directory = self.reader.directory
        else:
            self.directory = os.path.expanduser(directory)
            self.reader = sensor_for_directory(self.directory)

    @property
    def image_list(self):
        """List of scene metadata files (delegated to the sensor reader)."""
        return self.reader.image_list

    def get_pair_dict(self):
        """
        Get a dictionary with all stereo pair information for exactly two scenes.

        Creates a comprehensive dictionary containing stereo pair information,
        including convergence angle, base-to-height ratio, bisector elevation angle,
        asymmetry angle, and more.

        Returns
        -------
        dict
            Dictionary with stereo pair information and geometry parameters

        Raises
        ------
        ValueError
            If the inputs do not contain exactly two scenes. Use
            :meth:`get_pair_dicts` for the per-pair dictionaries of N scenes.
        """
        catid_dicts = self.get_catid_dicts()
        if len(catid_dicts) != 2:
            raise ValueError(
                f"get_pair_dict() requires exactly two scenes, but found "
                f"{len(catid_dicts)}. Use get_pair_dicts() for N-scene inputs."
            )
        catid1_dict, catid2_dict = catid_dicts
        pairname = os.path.split(self.directory.rstrip("/\\"))[-1]
        return self.pair_dict(catid1_dict, catid2_dict, pairname)

    def get_pair_dicts(self):
        """
        Get per-pair dictionaries for every combination of scenes.

        Builds one stereo-pair dictionary (see :meth:`pair_dict`) for each of the
        N-choose-2 combinations of the discovered scenes, so N-scene inputs can be
        assessed pairwise. For exactly two scenes this returns a single-element
        list; ``get_pair_dict`` remains the canonical two-scene entry point.

        Returns
        -------
        list of dict
            One stereo-pair dictionary per scene combination.

        Raises
        ------
        ValueError
            If fewer than two scenes are found (no pair can be formed).
        """
        catid_dicts = self.get_catid_dicts()
        if len(catid_dicts) < 2:
            raise ValueError(
                f"Need at least two scenes to form a stereo pair, but found "
                f"{len(catid_dicts)}."
            )
        base = os.path.split(self.directory.rstrip("/\\"))[-1]
        pairs = []
        for d1, d2 in combinations(catid_dicts, 2):
            c1 = d1.get("catid", "?")
            c2 = d2.get("catid", "?")
            pairname = f"{base}: {c1} / {c2}"
            pairs.append(self.pair_dict(d1, d2, pairname))
        return pairs

    def get_scenes_centroid_projection(self, proj_type="tmerc"):
        """
        Local projection centered on the union of all scene footprints.

        Used for the N-scene overview map so the projection is centered on all
        scenes together rather than a single pair intersection.

        Parameters
        ----------
        proj_type : str, optional
            Projection type, default "tmerc" (see :meth:`get_centroid_projection`).

        Returns
        -------
        str
            Proj4 string centered on the union of all scene footprints.
        """
        catid_dicts = self.get_catid_dicts()
        union = union_all([d["geom"] for d in catid_dicts])
        return self.get_centroid_projection(union, proj_type)

    def get_pair_map_projection(self, p, proj_type="tmerc"):
        """
        Local projection for a single pair's map, robust to no overlap.

        Centers on the pair intersection when the footprints overlap; otherwise
        falls back to the union of the two footprints so non-overlapping pairs
        (common in N-scene sets) still get a sensible map projection.

        Parameters
        ----------
        p : dict
            Stereo-pair dictionary from :meth:`pair_dict`.
        proj_type : str, optional
            Projection type, default "tmerc".

        Returns
        -------
        str
            Proj4 string centered on the pair's intersection or footprint union.
        """
        geom = p["intersection"]
        if geom is None:
            geom = union_all([p["catid1_dict"]["geom"], p["catid2_dict"]["geom"]])
        return self.get_centroid_projection(geom, proj_type)

    def get_catid_dicts(self):
        """
        Get dictionaries of metadata for each catalog ID.

        Delegates to the detected sensor reader to build a list of per-scene
        metadata dictionaries.

        Returns
        -------
        list
            List of dictionaries, one for each catalog ID, containing metadata
        """
        return self.reader.get_scene_dicts()

    def pair_dict(self, catid1_dict, catid2_dict, pairname):
        def center_date(dt_list):
            dt_list_sort = sorted(dt_list)
            dt_list_sort_rel = [dt - dt_list_sort[0] for dt in dt_list_sort]
            avg_timedelta = sum(dt_list_sort_rel, timedelta()) / len(dt_list_sort_rel)
            return dt_list_sort[0] + avg_timedelta

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

        p["conv_ang"] = get_convergence_angle(
            p["catid1_dict"]["meansataz"],
            p["catid1_dict"]["meansatel"],
            p["catid2_dict"]["meansataz"],
            p["catid2_dict"]["meansatel"],
        )

        p["bh"] = get_bh_ratio(p["conv_ang"])

        p["bie"] = get_bie_angle(
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
                from pyproj import Transformer

                centroid = p["intersection"].centroid
                transformer = Transformer.from_crs(
                    "EPSG:4326", "EPSG:4978", always_xy=True
                )
                # Transform (lon, lat, h=0) to ECEF (x, y, z)
                # h=0 places the point on the WGS84 ellipsoid surface
                ground_point = np.array(
                    transformer.transform(centroid.x, centroid.y, 0.0)
                )

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
        if intersection is not None:
            # Project to local CRS for area; skipped entirely when the footprints
            # do not overlap (common for some pairs of an N-scene set).
            intersection_local = geom2local(intersection)
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
