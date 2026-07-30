"""Generate the synthetic DIMAP v1 fixtures (SPOT 5, ALOS PRISM) in this directory.

Run as ``python make_fixtures.py tests/test_data/dimap_v1_synthetic`` to
regenerate ``spot5/`` and ``prism/``.

The layouts follow ASP's readers (SPOT_XML.cc, PRISM_XML.cc); the numbers are
synthetic but physically plausible (circular sun-synchronous orbit in ECEF).
"""

import math
import os
from datetime import datetime, timedelta

from pyproj import Transformer

MU = 3.986004418e14  # m^3/s^2
ECEF_TO_LLA = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)


def orbit(radius, inclination_deg, raan_deg, u0_deg, times_s):
    """Circular-orbit ECEF positions/velocities (Earth rotation ignored)."""
    i = math.radians(inclination_deg)
    om = math.radians(raan_deg)
    n = math.sqrt(MU / radius**3)
    out = []
    for t in times_s:
        u = math.radians(u0_deg) + n * t
        # In-orbit-plane coordinates, then rotate by inclination and RAAN.
        x0, y0 = radius * math.cos(u), radius * math.sin(u)
        vx0, vy0 = -radius * n * math.sin(u), radius * n * math.cos(u)
        xi, yi, zi = x0, y0 * math.cos(i), y0 * math.sin(i)
        vxi, vyi, vzi = vx0, vy0 * math.cos(i), vy0 * math.sin(i)
        x = xi * math.cos(om) - yi * math.sin(om)
        y = xi * math.sin(om) + yi * math.cos(om)
        vx = vxi * math.cos(om) - vyi * math.sin(om)
        vy = vxi * math.sin(om) + vyi * math.cos(om)
        out.append(((x, y, zi), (vx, vy, vzi)))
    return out


def iso(t0, dt):
    return (t0 + timedelta(seconds=dt)).strftime("%Y-%m-%dT%H:%M:%S.%f")


def ground_point(pos):
    lon, lat, _ = ECEF_TO_LLA.transform(*pos)
    return lon, lat


def frame_vertices(center_lon, center_lat, half=0.28):
    """Four scene corners, in the DIMAP v1 (row, col) corner order."""
    return [
        (center_lon - half, center_lat + half, 1, 1),
        (center_lon + half, center_lat + half, 1, 12000),
        (center_lon + half, center_lat - half, 12000, 12000),
        (center_lon - half, center_lat - half, 12000, 1),
    ]


def ephemeris_block(t0, step, states, indent):
    pad = " " * indent
    rows = []
    for k, ((x, y, z), (vx, vy, vz)) in enumerate(states):
        rows.append(
            f"{pad}<Point>\n"
            f"{pad}  <Location>\n"
            f"{pad}    <X>{x:.4f}</X>\n"
            f"{pad}    <Y>{y:.4f}</Y>\n"
            f"{pad}    <Z>{z:.4f}</Z>\n"
            f"{pad}  </Location>\n"
            f"{pad}  <Velocity>\n"
            f"{pad}    <X>{vx:.6f}</X>\n"
            f"{pad}    <Y>{vy:.6f}</Y>\n"
            f"{pad}    <Z>{vz:.6f}</Z>\n"
            f"{pad}  </Velocity>\n"
            f"{pad}  <TIME>{iso(t0, k * step)}</TIME>\n"
            f"{pad}</Point>"
        )
    return "\n".join(rows)


def vertex_block(vertices, indent):
    pad = " " * indent
    return "\n".join(
        f"{pad}<Vertex>\n"
        f"{pad}  <FRAME_LON>{lon:.6f}</FRAME_LON>\n"
        f"{pad}  <FRAME_LAT>{lat:.6f}</FRAME_LAT>\n"
        f"{pad}  <FRAME_ROW>{row}</FRAME_ROW>\n"
        f"{pad}  <FRAME_COL>{col}</FRAME_COL>\n"
        f"{pad}</Vertex>"
        for lon, lat, row, col in vertices
    )


def make_spot5(path, t0, raan, roll_rad, incidence, viewing, center, name):
    eph_step = 30.0
    states = orbit(7_200_000.0, 98.7, raan, 40.0, [k * eph_step for k in range(5)])
    center_lon, center_lat = center

    # Corrected attitude: yaw/pitch/roll in RADIANS (ASP feeds them straight to
    # sin/cos), off-nadir roll for an across-track HRG acquisition plus drift.
    att_step = 0.125
    angles = []
    for k in range(9):
        t = k * att_step
        angles.append(
            (
                iso(t0 + timedelta(seconds=25), t),
                0.0011 + 2.0e-6 * k,  # yaw
                -0.0025 + 5.0e-6 * k,  # pitch
                roll_rad + 1.0e-5 * k,  # roll
            )
        )
    angle_rows = "\n".join(
        "        <Angles>\n"
        f"          <TIME>{time}</TIME>\n"
        f"          <YAW>{yaw:.9f}</YAW>\n"
        f"          <PITCH>{pitch:.9f}</PITCH>\n"
        f"          <ROLL>{roll:.9f}</ROLL>\n"
        "        </Angles>"
        for time, yaw, pitch, roll in angles
    )

    xml = f"""<?xml version="1.0" encoding="ISO-8859-1"?>
<Dimap_Document>
  <Metadata_Id>
    <METADATA_FORMAT version="1.1">DIMAP</METADATA_FORMAT>
    <METADATA_PROFILE>SCENE_1A</METADATA_PROFILE>
    <METADATA_LANGUAGE>en</METADATA_LANGUAGE>
  </Metadata_Id>
  <Dataset_Id>
    <DATASET_NAME>{name}</DATASET_NAME>
    <COPYRIGHT>synthetic fixture, not a real delivery</COPYRIGHT>
  </Dataset_Id>
  <Dataset_Frame>
{vertex_block(frame_vertices(center_lon, center_lat), 4)}
    <SCENE_ORIENTATION>11.7</SCENE_ORIENTATION>
  </Dataset_Frame>
  <Raster_Dimensions>
    <NCOLS>12000</NCOLS>
    <NROWS>12000</NROWS>
    <NBANDS>1</NBANDS>
  </Raster_Dimensions>
  <Dataset_Sources>
    <Source_Information>
      <Scene_Source>
        <IMAGING_DATE>{t0.strftime("%Y-%m-%d")}</IMAGING_DATE>
        <IMAGING_TIME>{t0.strftime("%H:%M:%S")}</IMAGING_TIME>
        <MISSION>SPOT</MISSION>
        <MISSION_INDEX>5</MISSION_INDEX>
        <INSTRUMENT>HRG</INSTRUMENT>
        <INSTRUMENT_INDEX>1</INSTRUMENT_INDEX>
        <SENSOR_CODE>A</SENSOR_CODE>
        <SCENE_PROCESSING_LEVEL>1A</SCENE_PROCESSING_LEVEL>
        <INCIDENCE_ANGLE>{incidence}</INCIDENCE_ANGLE>
        <VIEWING_ANGLE>{viewing}</VIEWING_ANGLE>
        <SUN_AZIMUTH>151.882</SUN_AZIMUTH>
        <SUN_ELEVATION>42.316</SUN_ELEVATION>
        <THEORETICAL_RESOLUTION>2.5</THEORETICAL_RESOLUTION>
      </Scene_Source>
    </Source_Information>
  </Dataset_Sources>
  <Data_Strip>
    <Satellite_Time>
      <UT_DATE>{t0.strftime("%Y-%m-%d")}</UT_DATE>
    </Satellite_Time>
    <Ephemeris>
      <Points>
{ephemeris_block(t0, eph_step, states, 8)}
      </Points>
    </Ephemeris>
    <Satellite_Attitudes>
      <Corrected_Attitudes>
        <Corrected_Attitude>
{angle_rows}
        </Corrected_Attitude>
      </Corrected_Attitudes>
    </Satellite_Attitudes>
    <Sensor_Configuration>
      <Time_Stamp>
        <LINE_PERIOD>0.00075200</LINE_PERIOD>
        <SCENE_CENTER_TIME>{iso(t0, 30.0)}</SCENE_CENTER_TIME>
        <SCENE_CENTER_LINE>6000</SCENE_CENTER_LINE>
        <SCENE_CENTER_COL>6000</SCENE_CENTER_COL>
      </Time_Stamp>
    </Sensor_Configuration>
  </Data_Strip>
</Dimap_Document>
"""
    with open(path, "w") as f:
        f.write(xml)
    return angles, states, (center_lon, center_lat)


def make_prism(path, t0, u0, view, rpy0, center, name):
    eph_step = 10.0
    states = orbit(7_070_000.0, 98.16, 200.0, u0, [k * eph_step for k in range(6)])
    center_lon, center_lat = center

    # PRISM attitude angles are in DEGREES (ASP's rollPitchYaw converts them).
    att_step = 1.0
    angles = []
    for k in range(6):
        angles.append(
            (
                iso(t0, k * att_step),
                rpy0[0] + 0.0004 * k,  # roll
                rpy0[1] + 0.0002 * k,  # pitch
                rpy0[2] - 0.0003 * k,  # yaw
            )
        )
    angle_rows = "\n".join(
        "        <Angles>\n"
        f"          <TIME>{time}</TIME>\n"
        "          <Angle>\n"
        f"            <ROLL>{roll:.9f}</ROLL>\n"
        f"            <PITCH>{pitch:.9f}</PITCH>\n"
        f"            <YAW>{yaw:.9f}</YAW>\n"
        "          </Angle>\n"
        "        </Angles>"
        for time, roll, pitch, yaw in angles
    )

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Dimap_Document>
  <Metadata_Id>
    <METADATA_FORMAT version="1.1">DIMAP</METADATA_FORMAT>
    <METADATA_PROFILE>ALOS</METADATA_PROFILE>
    <METADATA_LANGUAGE>en</METADATA_LANGUAGE>
  </Metadata_Id>
  <Dataset_Id>
    <DATASET_NAME>{name}</DATASET_NAME>
    <COPYRIGHT>synthetic fixture, not a real delivery</COPYRIGHT>
  </Dataset_Id>
  <Dataset_Frame>
{vertex_block(frame_vertices(center_lon, center_lat, half=0.16), 4)}
  </Dataset_Frame>
  <Dataset_Sources>
    <Source_Information>
      <Scene_Source>
        <IMAGING_DATE>{t0.strftime("%Y-%m-%d")}</IMAGING_DATE>
        <IMAGING_TIME>{t0.strftime("%H:%M:%S")}</IMAGING_TIME>
        <MISSION>ALOS</MISSION>
        <MISSION_INDEX>1</MISSION_INDEX>
        <INSTRUMENT>{view}</INSTRUMENT>
        <INSTRUMENT_INDEX>1</INSTRUMENT_INDEX>
        <SUN_AZIMUTH>134.712</SUN_AZIMUTH>
        <SUN_ELEVATION>58.904</SUN_ELEVATION>
        <Image_Interpretation>
          <Spectral_Band_Info>
            <BAND_INDEX>1</BAND_INDEX>
            <NCOLS>4928</NCOLS>
            <NROWS>16000</NROWS>
          </Spectral_Band_Info>
        </Image_Interpretation>
      </Scene_Source>
    </Source_Information>
  </Dataset_Sources>
  <Data_Strip>
    <Satellite_Time>
      <TIME_FIRST_LINE>{iso(t0, 0.0)}</TIME_FIRST_LINE>
      <TIME_LAST_LINE>{iso(t0, 5.0)}</TIME_LAST_LINE>
    </Satellite_Time>
    <Ephemeris>
      <Points>
{ephemeris_block(t0, eph_step, states, 8)}
      </Points>
    </Ephemeris>
    <Satellite_Attitudes>
      <Angles_List>
{angle_rows}
      </Angles_List>
    </Satellite_Attitudes>
  </Data_Strip>
</Dimap_Document>
"""
    with open(path, "w") as f:
        f.write(xml)
    return angles, states, (center_lon, center_lat)


if __name__ == "__main__":
    import sys

    out_dir = sys.argv[1]
    os.makedirs(f"{out_dir}/spot5", exist_ok=True)
    os.makedirs(f"{out_dir}/prism", exist_ok=True)

    # Across-track SPOT 5 stereo pair over the same target: one scene rolled
    # ~+15 deg (east-looking), the other ~-12 deg (west-looking).
    spot_center = (26.816671, 43.098818)
    a, s, _ = make_spot5(
        f"{out_dir}/spot5/SPOT5_HRG1_SCENE_1A_east_synthetic.XML",
        datetime(2008, 3, 4, 12, 30, 33, 81912),
        35.0,
        0.2618,
        15.021,
        13.427,
        spot_center,
        "SPOT5 HRG1 SCENE 1A EAST SYNTHETIC",
    )
    print("SPOT5 east first angles (yaw,pitch,roll rad):", a[0])
    print("  -> roll deg:", math.degrees(a[0][3]), " pitch deg:", math.degrees(a[0][2]))
    print("SPOT5 radius:", math.dist(s[0][0], (0, 0, 0)))
    make_spot5(
        f"{out_dir}/spot5/SPOT5_HRG2_SCENE_1A_west_synthetic.XML",
        datetime(2008, 3, 11, 12, 34, 12, 415003),
        38.5,
        -0.2094,
        12.043,
        10.781,
        spot_center,
        "SPOT5 HRG2 SCENE 1A WEST SYNTHETIC",
    )

    # PRISM forward/backward pair of a triplet over the same target.
    prism_center = (33.181672, 57.994491)
    a, s, _ = make_prism(
        f"{out_dir}/prism/PRISM_ALOS_forward_synthetic.XML",
        datetime(2007, 5, 19, 1, 23, 45, 500000),
        120.0,
        "PRISM FORWARD",
        (0.0521, -0.0312, 0.1187),
        prism_center,
        "ALPSMF_SYNTHETIC_FORWARD",
    )
    print("PRISM forward first angles (roll,pitch,yaw deg):", a[0])
    print("PRISM radius:", math.dist(s[0][0], (0, 0, 0)))
    make_prism(
        f"{out_dir}/prism/PRISM_ALOS_backward_synthetic.XML",
        datetime(2007, 5, 19, 1, 24, 30, 250000),
        120.35,
        "PRISM BACKWARD",
        (0.0489, -0.0298, 0.1203),
        prism_center,
        "ALPSMB_SYNTHETIC_BACKWARD",
    )
