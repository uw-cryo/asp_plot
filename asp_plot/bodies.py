"""Per-body facts for the planetary bodies asp_plot supports.

Earth, the Moon, and Mars each carry a handful of constants that the rest of
the package needs: an altimetry instrument name, an IAU sphere radius, a
``pc_align --datum`` string, a geocentric PROJ string, and a geographic CRS
WKT.  Historically these were re-typed as ad-hoc ``{"moon": ..., "mars": ...}``
dict literals at every use site.  This module collects them into one frozen
:class:`Body` dataclass and a :data:`BODIES` registry so that adding a body (or
correcting a constant) is a single-line change rather than a grep across files.

The canonical body *detector* remains :func:`asp_plot.utils.detect_planetary_body`,
which inspects a DEM's CRS WKT.  :func:`body_for_dem` wraps it to return the
matching :class:`Body`.
"""

from dataclasses import dataclass

# Geographic CRS WKT strings for building planetary GeoDataFrames. Kept as
# module constants so the (long) WKT lives in exactly one place.
_MOON_GEO_CRS = 'GEOGCRS["Moon",DATUM["D_MOON",ELLIPSOID["MOON",1737400,0]],PRIMEM["Reference_Meridian",0],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]]]'
_MARS_GEO_CRS = 'GEOGCRS["Mars",DATUM["D_MARS",ELLIPSOID["MARS",3396190,0]],PRIMEM["Reference_Meridian",0],CS[ellipsoidal,2],AXIS["latitude",north,ORDER[1],ANGLEUNIT["degree",0.0174532925199433]],AXIS["longitude",east,ORDER[2],ANGLEUNIT["degree",0.0174532925199433]]]'


@dataclass(frozen=True)
class Body:
    """Immutable bundle of per-body facts.

    Attributes
    ----------
    name : str
        Canonical body name: ``"earth"``, ``"moon"``, or ``"mars"``.  Matches
        the strings returned by :func:`asp_plot.utils.detect_planetary_body`.
    altimetry_instrument : str
        Reference altimetry instrument: ``"ICESat-2"``, ``"LOLA"``, or
        ``"MOLA"``.
    iau_sphere_radius_m : float or None
        IAU mean sphere radius in meters, used to convert planetary radius to
        height above the sphere.  ``None`` for Earth, which uses an ellipsoid
        rather than a sphere.
    datum : str or None
        ASP ``pc_align --datum`` string (``"D_MOON"`` / ``"D_MARS"``).
        ``None`` for Earth (pc_align infers the datum from the data).
    geocentric_proj : str or None
        Body-centered geocentric ("ECEF-equivalent") PROJ string used to
        convert a Cartesian translation vector into the DEM's CRS.  ``None``
        for Earth, where ``EPSG:4978`` is used instead (PROJ refuses to
        operate across celestial bodies, so planets need an explicit string).
    geographic_crs_wkt : str or None
        Geographic CRS WKT used when building planetary GeoDataFrames.
        ``None`` for Earth, whose data already carry a CRS.
    semi_major_axis_m : float
        Ellipsoid semi-major axis in meters (fallback for building a
        geographic CRS when a DEM's own ellipsoid is unavailable).
    inverse_flattening : float
        Ellipsoid inverse flattening (``0.0`` for the spherical bodies).
    """

    name: str
    altimetry_instrument: str
    iau_sphere_radius_m: "float | None"
    datum: "str | None"
    geocentric_proj: "str | None"
    geographic_crs_wkt: "str | None"
    semi_major_axis_m: float
    inverse_flattening: float


BODIES = {
    "earth": Body(
        name="earth",
        altimetry_instrument="ICESat-2",
        iau_sphere_radius_m=None,
        datum=None,
        geocentric_proj=None,  # use EPSG:4978
        geographic_crs_wkt=None,
        semi_major_axis_m=6378137.0,
        inverse_flattening=298.257223563,
    ),
    "moon": Body(
        name="moon",
        altimetry_instrument="LOLA",
        iau_sphere_radius_m=1_737_400.0,
        datum="D_MOON",
        geocentric_proj="+proj=geocent +R=1737400 +units=m +no_defs",
        geographic_crs_wkt=_MOON_GEO_CRS,
        semi_major_axis_m=1_737_400.0,
        inverse_flattening=0.0,
    ),
    "mars": Body(
        name="mars",
        altimetry_instrument="MOLA",
        iau_sphere_radius_m=3_396_190.0,
        datum="D_MARS",
        geocentric_proj="+proj=geocent +R=3396190 +units=m +no_defs",
        geographic_crs_wkt=_MARS_GEO_CRS,
        semi_major_axis_m=3_396_190.0,
        inverse_flattening=0.0,
    ),
}


def body_for_dem(dem_fn, body=None):
    """Return the :class:`Body` for a DEM.

    Parameters
    ----------
    dem_fn : str
        Path to the DEM file.
    body : str or None, optional
        Body name to look up directly.  If ``None`` (default), the body is
        auto-detected from the DEM's CRS via
        :func:`asp_plot.utils.detect_planetary_body`.

    Returns
    -------
    Body
    """
    # Imported lazily to avoid a circular import (utils imports nothing from
    # here, but keeping the import local keeps this module dependency-light).
    from asp_plot.utils import detect_planetary_body

    if body is None:
        body = detect_planetary_body(dem_fn)
    return BODIES[body]
