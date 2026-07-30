# Plan: broaden sensor metadata support (#168)

**Status: proposal for discussion — this PR contains only this document.**
Implementation would land as the follow-up PRs sequenced at the bottom and would
close #161, #162, and #163 along the way. This file is removed once the plan is
agreed and tracked in the implementation PRs.

## 1. The question #168 asks

Should we translate ASP's own camera-XML readers into `asp_plot` readers *up
front*, without real sample data, and let users break them and report issues?

**Short answer: yes for one family, no as a blanket policy.** The full set of
Earth linescan formats ASP reads directly is closed and small, and it splits
cleanly by risk:

| Format (ASP session) | ASP reader | What the file carries | asp_plot today | Verdict |
|---|---|---|---|---|
| DigitalGlobe / WorldView (`dg`) | `RPC_XML.cc`, `LinescanDGModel.cc` | ephemeris + tabulated quaternions + covariances, full angle/TDI/scandir summary | supported | harden detection & degradation (#162, #163) |
| Pléiades Neo (`pleiades`) | `PleiadesXML.cc` (`PNEO_SENSOR`) | ephemeris + tabulated quaternions, `Located_Geometric_Values` angle grid | supported | — |
| Pléiades 1A/1B (`pleiades`) | `PleiadesXML.cc` (`PHR_SENSOR`) | same, but attitude is a **degree-3 `Polynomial_Quaternions`** block | attitude silently empty (#161) | implement from ASP's spec |
| SPOT 6/7 (`pleiades`) | `PleiadesXML.cc` (`S6_SENSOR`/`S7_SENSOR`) | same layout as Neo; time spacing may be non-uniform | likely parses by accident, unlabelled/untested | gate + name + test explicitly |
| PeruSat-1 (`perusat`) | `PeruSatXML.cc` (`PER1_SENSOR`) | same DIMAP v2 layout: `Refined_Model/{Ephemeris/Point_List, Attitudes/Quaternion_List}`, scalar-first `Q0`, single `Located_Geometric_Values` | unsupported | small profile gate on the DIMAP reader |
| SPOT 5 (`spot5`) | `SPOT_XML.cc` | DIMAP v1: ephemeris `Points`, attitude as **tabulated roll/pitch/yaw** (`Corrected_Attitudes`), `Dataset_Frame` corners; no angle-summary grid | unsupported | second tier — needs an attitude-representation decision (§3.2) |
| ALOS PRISM (`prism`) | `PRISM_XML.cc` (`METADATA_PROFILE = ALOS`) | DIMAP-adjacent: ephemeris `Points`, attitude as roll/pitch/yaw `Angles_List` | unsupported | second tier, same decision |
| ASTER (`aster`) | `ASTER_XML.cc` | **only** lattice points, sight vectors, satellite positions, image size — no times, no attitude, no view/sun angles, no footprint corners | date parsed from `AST_L1A_*` filename (`utils.get_acquisition_dates`) | **not planned** (§4) |
| Planetary (`csm`, ISIS cubes) | `CsmModel.cc` etc. | CSM state | handled by `csm_camera.py`/`csm_io.py` | out of scope for `sensors/` |
| Generic RPC, pinhole, optical-bar (KH-9) | — | no ephemeris/attitude metadata to plot | n/a | out of scope |

Two structural facts make the "up front" idea much cheaper than it sounds:

1. **Everything worth adding is one DIMAP family.** Pléiades 1A/1B, Neo,
   SPOT 6/7, and PeruSat-1 share the `Dimap_Document` root, the
   `Refined_Model` ephemeris/attitude layout, and scalar-first quaternions —
   ASP itself handles PHR/PNEO/S6/S7 in a single reader gated on
   `METADATA_PROFILE`. We are not writing N speculative parsers; we are adding
   profile gates and one attitude variant to a reader we already validated
   against real Pléiades Neo data (PR #155's audit).
2. **The blast radius of a wrong parser is small and visible.** The scene
   dicts feed exactly one consumer chain (`stereopair_metadata_parser` →
   `stereo_geometry`), the report pipeline already wraps it in
   `try/except ValueError` with a graceful fallback, and the plots render NaN
   summary values as `nan` rather than crashing. A bad speculative reader
   produces a wrong geometry plot, not a broken report.

The real risk is not "parser is wrong" but "speculative reader claims files it
shouldn't and shadows a working one." That makes #162 (content-based
detection) a **prerequisite** of the expansion, not incidental hardening.

## 2. What the consumer audit says we actually need

Traced every scene-dict key through the codebase:

- **Drive the plots** (must be right): `catid`, `date`, `geom`, `meansataz`,
  `meansatel`, `eph_gdf`, `att_df`, `fp_gdf`; the remaining view-angle/GSD
  means appear in scene strings only.
- **Already optional by convention**: `scandir`/`tdi` (omitted from scene
  strings when `None` — the Pléiades path established this), all `cov_*`
  columns (plots annotate "covariance not provided"), `asymmetry_angle`
  (renders "N/A").
- **No production consumer at all**: `xml_fn`, `sensor`, `meansunaz`,
  `meansunel`, `cloudcover` — used only in tests and the scene-selection
  notebooks. New readers should still fill them (cheap, and the notebooks are
  a supported use), but they are not a reason to block a sensor.
- **Gaps in the degradation story** (fix in Tier 0): `WorldViewMetadata`
  hard-crashes on a missing `TDILEVEL`/summary tag (#163);
  `get_pair_utm_epsg()`/`get_intersection_bounds()` crash on a
  non-overlapping pair while the neighboring code handles `None`
  intersections; `pair_dict()` assumes `date` is never None.

Conclusion: the sensor-agnostic scene-dict schema is right, but it should be
**formalized as required-core + optional-blocks** instead of "whatever the two
existing readers happen to produce":

- *identity core* (required): `catid`, `sensor`, `date`, `geom`, `xml_fn`
- *summary block* (optional, defaults NaN/None): the mean-angle/GSD/sun/cloud
  fields, `scandir`, `tdi`
- *trajectory block* (optional, when `geteph=True`): `eph_gdf`, `att_df`,
  `fp_gdf`

A `base.py` helper fills the defaults so each reader states only what its
format provides, and consumers rely on one documented "not provided"
convention rather than per-key accidents.

## 3. Design decisions

### 3.1 Package layout: `sensors.py` → `sensors/`

Yes to the subpackage — `sensors.py` is 1,166 lines with two readers; the
DIMAP work adds material to only one of them, and second-tier readers would
each bring a few hundred lines more.

```
asp_plot/sensors/
    __init__.py     # SENSORS registry, sensor_for_directory/inputs,
                    # resolve_xml_inputs, and re-exports of every current
                    # public name (SensorMetadata, WorldViewMetadata,
                    # PleiadesMetadata, ...)
    base.py         # SensorMetadata ABC, scene-dict schema + default-filling
                    # helper, _common_base, shared XML discovery helpers
    worldview.py    # WorldViewMetadata (incl. dg_mosaic handling)
    dimap.py        # Airbus/ADS DIMAP v2 family: Pléiades 1A/1B + Neo,
                    # SPOT 6/7, PeruSat-1 — one reader, METADATA_PROFILE-gated
    spot5.py        # tier 2 (DIMAP v1)
    prism.py        # tier 2
```

`from asp_plot.sensors import X` keeps working for every current public name
via `__init__.py` re-exports — `stereopair_metadata_parser.py` (the only
production importer) and the tests don't change. **This is why none of this is
a v3: it's additive readers plus an internal reorganization behind a stable
import path. It's v2.x.**

### 3.2 Attitude comes in three shapes — make the contract explicit

Across the table in §1, attitude is: tabulated quaternions (WV, Neo, SPOT 6/7,
PeruSat), *polynomial* quaternions (Pléiades 1A/1B), tabulated
roll/pitch/yaw (SPOT 5, PRISM), or absent (ASTER). Current code assumes shape
one (`att_df` with `q1..q4`; `_compute_roll_pitch_yaw` converts to LVLH RPY).

Contract: `att_df` is time-indexed and carries **either** `q1..q4` (+
`cov_*`) **or** `roll/pitch/yaw` columns, or is `None`. The orientation
plotting grows a small dispatch: quaternions → compute LVLH RPY as today;
native RPY → plot directly (labelled with the file's own frame, since the
DIMAP v1/PRISM angles are already satellite-frame attitude angles — verify
frame semantics before claiming comparability with the LVLH panel); `None` →
the existing "not provided" annotation. Polynomial quaternions are a *reader*
concern: evaluate the degree-3 polynomial per component at the ephemeris
timestamps (exactly ASP's `read_attitudes_1A1B` + `get_camera_pose_at_time`
recipe) and emit ordinary tabulated `q1..q4` — downstream never knows.

### 3.3 Detection must be content-based before the registry grows (#162)

Today WorldView claims *any* non-ortho/README XML, tolerable only because the
registry has two entries and Pléiades detects strictly first. With more DIMAP
profiles in play:

- WorldView: require the DG blocks (root `<isd>` and/or `EPH`/`ATT`/`IMD`)
  via the same cheap `iterparse` peek `_is_dimap_product()` already uses.
  Verify `dg_mosaic` `.r100.xml` outputs still pass (our fixtures retain the
  blocks).
- DIMAP: dispatch on `METADATA_PROFILE` (`PHR_SENSOR`, `PNEO_SENSOR`,
  `S6_SENSOR`, `S7_SENSOR`, `PER1_SENSOR`) instead of accepting any
  `Dimap_Document` with subprofile `PRODUCT`. Unknown profiles get a clear
  "DIMAP product with unsupported profile X" message instead of a deep parse
  error.
- Unrecognized inputs then fall through to the existing clean "no supported
  sensor found" error.

### 3.4 Spec-based readers are labelled, not silent

For any reader (or reader branch) written from ASP's source rather than
validated against real data:

- a `validated_against_real_data` class flag; when False, `get_scene_dicts()`
  logs a one-time warning naming the ASP reader it mirrors and linking to the
  issue tracker — this operationalizes #168's "let users break it and report";
- a support-matrix table in the docs (sensor, session type, status:
  *validated* / *implemented from ASP spec* / *not planned*), which also
  resolves #163's "document the DIMAP V2 scope" item;
- tests run on **synthetic fixtures generated from ASP's reader structure**
  (the trimmed 8-point Pléiades Neo fixtures, 68 KB total, are the template —
  small, hand-checkable, committed). Where ASP's own test data or docs give
  numeric examples (e.g. the 1A/1B polynomial evaluation), pin those numbers.

## 4. What we deliberately don't do

- **No ASTER reader.** The ASP ASTER XML carries no timestamps, no attitude,
  no view/sun-angle summary, and no footprint corners — nearly every panel of
  the geometry plots would be blank. ASTER users already get reports (the
  geometry section degrades gracefully; acquisition dates come from the
  `AST_L1A_*` filename). If demand appears, the honest feature is a
  positions-only trajectory plot, and that's a separate issue with its own
  design.
- **No planetary sensors in `sensors/`.** HiRISE/CTX/LRO-NAC/etc. enter ASP
  via ISIS/CSM; `csm_camera.py` is the right home and already handles them.
- **No generic-RPC or optical-bar (KH-9) readers** — the formats carry no
  trajectory/attitude metadata to plot.
- **No new hard dependencies.** Everything here is `xml.etree` + numpy/pandas
  parsing, same as today.

## 5. Implementation sequence

Small PRs, each independently shippable; expected releases are v2.x minors.

1. **PR A — this plan** (docs only, discussion).
2. **PR B — package split** (pure move: `sensors.py` → `sensors/` with
   `__init__` re-exports; zero behavior change; incidental: fix the stale
   `parser.get_id_dict(...)` calls in the two scene-selection notebooks, which
   broke when readers were extracted from the parser).
3. **PR C — detection + degradation hardening** (*closes #162, #163*):
   content-based WorldView detection; `METADATA_PROFILE` dispatch;
   optional-TDI/summary-tag handling in the WorldView reader; the
   consumer-side None/NaN gaps from §2; scene-dict schema contract documented
   in `base.py` + ARCHITECTURE.md; docs support matrix.
4. **PR D — DIMAP family completion** (*closes #161, and #168 with it*):
   1A/1B `Polynomial_Quaternions` evaluated to tabulated quaternions;
   SPOT 6/7 explicitly gated, named (`SPOT6`/`SPOT7` sensor string), and
   tested; PeruSat-1 profile gate (scalar-first quats, single
   `Located_Geometric_Values`). All spec-based branches carry the §3.4
   labelling. Synthetic fixtures for each profile.
5. **PR E (optional, demand-driven) — SPOT 5 + PRISM**: new `spot5.py` /
   `prism.py` readers with the native-RPY attitude path from §3.2. Only worth
   opening if a user shows up with data — but after PR C/D the marginal cost
   is one module each, and the attitude contract is already in place.

## 6. Open questions for discussion

1. **Frame semantics of native RPY** (SPOT 5 / PRISM): confirm those angles
   are comparable to (or clearly labelled as distinct from) the LVLH
   roll/pitch/yaw we compute from quaternions, before they share a panel.
2. **Is PeruSat/SPOT 6/7/PRISM demand real?** The tiering keeps us honest —
   PR D is cheap because it rides the validated DIMAP reader; PR E waits for
   a user.
3. **Naming**: does `dimap.py`/"DIMAP family" read better than keeping the
   `PleiadesMetadata` class name with profile gates? Suggest: keep the class
   name (public API stability), house it in `dimap.py`, and report `sensor`
   strings per profile (`Pleiades1A`, `PNEO3`, `SPOT6`, `PeruSat1`, ...).
