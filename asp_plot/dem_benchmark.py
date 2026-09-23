"""
Benchmark many DEMs against one altimetry reference (issue #169).

The report scores *one* DEM against ICESat-2 (or LOLA/MOLA). This module
scores *any number* of DEMs -- different scene combinations, joint multi-view
triangulation vs. pairwise stereo + ``dem_mosaic``, parameter sweeps -- against
the same fixed altimetry sample, so the numbers are directly comparable:

- coverage inside a common area of interest (by default the intersection of
  all DEM footprints, so crop windows that differ per run compare fairly)
- triangulation error, from the ``*-IntersectionErr.tif`` ``point2dem`` writes
  next to each ``*-DEM.tif`` (absent for a mosaic), in the table only: it is
  a consistency check on a run, not a quality ranking across geometries
- altimetry residuals (altimetry minus DEM: n, median, NMAD, RMSE) before and,
  optionally, after a ``pc_align --compute-translation-only`` per DEM, with the
  translation that removed -- on each DEM's own points and on the points
  every DEM shares
- a paired block bootstrap of the shared-point statistics: 95 % intervals on
  each DEM's median and NMAD, on its NMAD difference to the best DEM, and the
  probability that it ranks first (issue #199)
- optionally, each DEM's difference against one of the candidates named as
  the reference (DEM minus reference median / NMAD)

Every DEM is scored with the same recipe the report uses for one DEM: the
cached ATL06-SR parquet is replayed (no SlideRule request), water returns are
dropped with the ESA WorldCover classes stored in the cache, and dh outliers
beyond 3σ are removed per DEM (after a gross-outlier cut at 30 NMAD). ``pc_align`` products and translated DEM copies
are kept out of the candidates' folders, under
``<directory>/dem_benchmark/<label>/``, so nothing is written into the stereo runs.

Each DEM keeps a different set of points -- its voids differ and the outlier
cut runs per DEM -- so the per-DEM numbers are not a fair ranking: a blend
that fills voids is scored on harder ground than a single pair that skips it.
The headline statistics (``*_shared_*`` columns, the figure, the sort order)
are therefore computed on the points valid in every DEM. Residuals are also
spatially correlated along an ICESat-2 beam (40 m segments posted every 20 m,
plus terrain and land cover), so the bootstrap resamples whole beam tracks,
not points, and the DEMs are compared through their paired differences on
the same replicates rather than through overlapping per-DEM intervals.

Examples
--------
>>> from asp_plot.dem_benchmark import DEMBenchmark
>>> bench = DEMBenchmark(
...     directory="atlanta_mvs",
...     dems={
...         "MVS 3-scene": "atlanta_mvs/stereo_mvs3/run-DEM.tif",
...         "MVS 5-scene": "atlanta_mvs/stereo_mvs5/run-DEM.tif",
...         "3 pairs + mosaic": "atlanta_mvs/pairwise_mosaic-DEM.tif",
...     },
...     parquet="atlanta_mvs/atl06sr_all.parquet",
... )
>>> stats = bench.run()
>>> bench.summary_plot(save_dir="atlanta_mvs", fig_fn="dem_benchmark.png")
"""

import logging
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.errors import WindowError
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds, intersection

from asp_plot.alignment import Alignment
from asp_plot.altimetry import Altimetry
from asp_plot.utils import Raster, detect_planetary_body, glob_file, nmad, save_figure

logger = logging.getLogger(__name__)

#: Cap on pixels read per DEM window; larger windows are read downsampled
#: (nearest), which keeps a 1 m, 20k x 20k DEM at ~16 M samples. The valid
#: fraction is then a subsample estimate; the area is scaled from it.
MAX_WINDOW_PIXELS = 16_000_000

#: ASP-style generic output prefixes, for which the folder name is the better
#: label (``stereo_mvs3/run-DEM.tif`` -> ``stereo_mvs3``).
GENERIC_STEMS = {"run", "out", "output", "dem"}

#: Stats columns, in dataframe order; the aligned/vs-reference groups are NaN
#: when pc_align is off / no reference is named.
STATS_COLUMNS = [
    "label",
    "dem_fn",
    "gsd_m",
    "valid_pct",
    "valid_area_km2",
    "ie_median_m",
    "ie_nmad_m",
    "n_points",
    "dh_median_m",
    "dh_nmad_m",
    "dh_rmse_m",
    "translation_m",
    "north_shift_m",
    "east_shift_m",
    "down_shift_m",
    "dh_aligned_median_m",
    "dh_aligned_nmad_m",
    "dh_aligned_rmse_m",
    "n_shared",
    "dh_shared_median_m",
    "dh_shared_nmad_m",
    "dh_shared_rmse_m",
    "dh_aligned_shared_median_m",
    "dh_aligned_shared_nmad_m",
    "dh_aligned_shared_rmse_m",
    "median_ci_low_m",
    "median_ci_high_m",
    "nmad_ci_low_m",
    "nmad_ci_high_m",
    "nmad_vs_best_ci_low_m",
    "nmad_vs_best_ci_high_m",
    "p_best",
    "vs_ref_median_m",
    "vs_ref_nmad_m",
]

_ALIGNED_KEYS = (
    "translation_m",
    "north_shift_m",
    "east_shift_m",
    "down_shift_m",
    "dh_aligned_median_m",
    "dh_aligned_nmad_m",
    "dh_aligned_rmse_m",
)

_SHARED_KEYS = (
    "n_shared",
    "dh_shared_median_m",
    "dh_shared_nmad_m",
    "dh_shared_rmse_m",
    "dh_aligned_shared_median_m",
    "dh_aligned_shared_nmad_m",
    "dh_aligned_shared_rmse_m",
)

_BOOTSTRAP_KEYS = (
    "median_ci_low_m",
    "median_ci_high_m",
    "nmad_ci_low_m",
    "nmad_ci_high_m",
    "nmad_vs_best_ci_low_m",
    "nmad_vs_best_ci_high_m",
    "p_best",
)

#: Column added to the altimetry point tables to identify one point across
#: the per-DEM ``Altimetry`` objects (the tables share a non-unique time index).
POINT_ID = "benchmark_point_id"

#: ATL06-SR columns identifying one beam track: a resampling block.
ICESAT2_BLOCK_COLUMNS = ("rgt", "cycle", "spot")

#: Candidate track-id columns in a LOLA/MOLA CSV (matched stripped, lowercase).
#: MOLA PEDR exports carry ``ORBIT``; LOLA RDR exports carry none, and the
#: bootstrap then resamples points.
PLANETARY_BLOCK_COLUMNS = ("orbit", "orbit_number", "orbitnumber", "track", "track_id")


def label_from_dem_path(dem_fn):
    """
    Derive a short label for a DEM from its path.

    Strips the ASP ``-DEM.tif`` suffix; when what remains is a generic ASP
    output prefix (``run``, ``out``, ...) the containing folder is used
    instead, since that is what distinguishes one run from another.

    Examples
    --------
    >>> label_from_dem_path("atlanta_mvs/stereo_mvs3/run-DEM.tif")
    'stereo_mvs3'
    >>> label_from_dem_path("atlanta_mvs/pairwise_mosaic-DEM.tif")
    'pairwise_mosaic'
    """
    stem = os.path.basename(dem_fn)
    for suffix in ("-DEM.tif", "-DEM.tiff", ".tif", ".tiff"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parent = os.path.basename(os.path.dirname(os.path.abspath(dem_fn)))
    if stem.lower() in GENERIC_STEMS and parent:
        return parent
    return stem


def parse_dem_specs(specs):
    """
    Turn CLI-style DEM specs into an ordered ``{label: path}`` mapping.

    Each spec is either a path (labelled via :func:`label_from_dem_path`) or
    ``label=path``. Duplicate labels are disambiguated with the containing
    folder so two ``run-DEM.tif`` never collapse into one row.
    """
    dems = {}
    for spec in specs:
        if "=" in spec and not os.path.exists(spec):
            label, path = spec.split("=", 1)
            label = label.strip()
        else:
            label, path = label_from_dem_path(spec), spec
        if label in dems:
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            candidate = f"{parent}/{label}"
            n = 2
            while candidate in dems:
                candidate = f"{parent}/{label} ({n})"
                n += 1
            label = candidate
        dems[label] = path
    return dems


def intersection_error_path(dem_fn):
    """Path of the ``point2dem --errorimage`` sibling of ``<prefix>-DEM.tif``, or None."""
    for suffix in ("-DEM.tif", "-DEM.tiff"):
        if dem_fn.endswith(suffix):
            candidate = (
                dem_fn[: -len(suffix)] + "-IntersectionErr" + suffix[-len(".tif") :]
            )
            if suffix.endswith(".tiff"):
                candidate = dem_fn[: -len(suffix)] + "-IntersectionErr.tiff"
            return candidate if os.path.exists(candidate) else None
    return None


def _safe_dirname(label):
    """Filesystem-safe folder name for a label (``MVS 3-scene`` -> ``MVS_3-scene``)."""
    return re.sub(r"[^\w.-]+", "_", label).strip("_") or "dem"


def _dh_stats(dh, prefix):
    """median / NMAD / RMSE of a dh series as ``{prefix}_*_m`` keys (NaN when empty)."""
    values = np.asarray(dh, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_median_m": np.nan,
            f"{prefix}_nmad_m": np.nan,
            f"{prefix}_rmse_m": np.nan,
        }
    return {
        f"{prefix}_median_m": float(np.median(values)),
        f"{prefix}_nmad_m": float(nmad(values)),
        f"{prefix}_rmse_m": float(np.sqrt(np.mean(values**2))),
    }


def _row_nmad(values):
    """NMAD of each row of a 2-D array (NaN-aware)."""
    med = np.nanmedian(values, axis=1, keepdims=True)
    return 1.4826 * np.nanmedian(np.abs(values - med), axis=1)


def paired_block_bootstrap(values, blocks=None, n_boot=1000, seed=0):
    """
    Bootstrap the median and NMAD of several residual series scored on the
    same points, resampling blocks of points and keeping the series paired.

    Parameters
    ----------
    values : array-like, shape (n_series, n_points)
        Residuals of each series (DEM) at the same points. NaN is allowed
        and ignored.
    blocks : array-like of length n_points, optional
        Block label of each point (an ICESat-2 beam track, a MOLA orbit).
        Whole blocks are drawn with replacement; None resamples points,
        which treats them as independent and gives optimistic intervals
        when residuals are spatially correlated.
    n_boot : int, optional
        Number of replicates, default 1000.
    seed : int, optional
        Seed of the random generator, default 0.

    Returns
    -------
    dict
        ``median`` and ``nmad``: arrays of shape (n_boot, n_series) with the
        statistic of every series on each replicate; ``n_blocks``: the
        number of blocks resampled (the number of points when ``blocks`` is
        None).

    Notes
    -----
    Every series is evaluated on the same replicate, so a difference between
    two series computed replicate by replicate is a paired estimate: the
    shared sampling variation cancels, which is what makes DEMs scored on the
    same points separable even when their own intervals overlap.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be 2-D: (n_series, n_points).")
    n_points = values.shape[1]
    if blocks is None:
        groups = [np.array([i]) for i in range(n_points)]
    else:
        codes, _ = pd.factorize(np.asarray(blocks))
        order = np.argsort(codes, kind="stable")
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        groups = np.split(order, bounds)
    n_blocks = len(groups)
    rng = np.random.default_rng(seed)
    medians = np.full((n_boot, values.shape[0]), np.nan)
    nmads = np.full((n_boot, values.shape[0]), np.nan)
    for b in range(n_boot):
        chosen = rng.integers(0, n_blocks, size=n_blocks)
        idx = np.concatenate([groups[g] for g in chosen])
        sample = values[:, idx]
        medians[b] = np.nanmedian(sample, axis=1)
        nmads[b] = _row_nmad(sample)
    return {"median": medians, "nmad": nmads, "n_blocks": n_blocks}


def _window_for_bounds(ds, bounds, bounds_crs):
    """
    Pixel window of ``ds`` covering ``bounds`` (in ``bounds_crs``), clipped to
    the raster. None when the two do not overlap; the full raster when
    ``bounds`` is None.
    """
    full = Window(0, 0, ds.width, ds.height)
    if bounds is None:
        return full
    if bounds_crs is not None and ds.crs is not None and bounds_crs != ds.crs:
        bounds = transform_bounds(bounds_crs, ds.crs, *bounds)
    win = from_bounds(*bounds, transform=ds.transform)
    win = win.round_offsets().round_lengths()
    try:
        win = intersection(win, full)
    except WindowError:
        return None
    if win.width < 1 or win.height < 1:
        return None
    return win


def _read_window(ds, win):
    """Masked read of one band inside ``win``, downsampled past MAX_WINDOW_PIXELS."""
    npix = float(win.width) * float(win.height)
    if npix > MAX_WINDOW_PIXELS:
        factor = np.sqrt(npix / MAX_WINDOW_PIXELS)
        out_shape = (
            max(1, int(round(win.height / factor))),
            max(1, int(round(win.width / factor))),
        )
        a = ds.read(1, window=win, out_shape=out_shape, masked=True)
    else:
        a = ds.read(1, window=win, masked=True)
    return np.ma.masked_invalid(a)


class DEMBenchmark:
    """
    Score many DEMs against one altimetry sample.

    Parameters
    ----------
    directory : str
        Working folder. pc_align products, the pc_align CSV and the translated
        DEM copy for each candidate go under ``<directory>/dem_benchmark/<label>/``.
        The translated copy is rewritten on every run, so two benchmarks
        sharing a directory and a label must not run at the same time.
    dems : dict or list
        ``{label: dem_fn}`` (order preserved), or a list of paths /
        ``"label=path"`` strings (see :func:`parse_dem_specs`).
    parquet : str or dict, optional
        ICESat-2 ATL06-SR parquet cache written by a prior report or
        ``Altimetry.request_atl06sr_multi_processing(save_to_parquet=True)``;
        either one path (used as processing level ``key``) or a
        ``{key: path}`` mapping. Required for Earth DEMs.
    altimetry_csv : str, optional
        LOLA/MOLA CSV (see ``request_planetary_altimetry``). Required for
        Moon/Mars DEMs.
    key : str, optional
        ATL06-SR processing level to score against, default ``"all"``.
    reference : str, optional
        Label of the candidate the others are differenced against for the
        ``vs_ref_*`` columns. Default None (columns left NaN).
    aoi : str, tuple or None, optional
        Area for the coverage and triangulation-error statistics:
        ``"intersection"`` (default) uses the common footprint of all DEMs;
        a ``(left, bottom, right, top)`` tuple is taken in the first DEM's
        CRS; None uses each DEM's own extent (not comparable across crops).
    filter_out : str or None, optional
        ESA WorldCover group dropped before differencing, default
        ``"water"`` (the report's setting). None keeps every return.
    n_sigma : float or None, optional
        Per-DEM dh outlier cut, default 3 (the report's setting).
    n_bootstrap : int, optional
        Replicates of the paired block bootstrap on the shared-point
        statistics, default 1000. 0 skips it (the ``*_ci_*`` and ``p_best``
        columns stay NaN).
    seed : int, optional
        Seed of the bootstrap's random generator, default 0.
    title : str, optional
        Figure title.

    Notes
    -----
    After :meth:`run`, ``stats_df`` holds one row per DEM with the columns in
    :data:`STATS_COLUMNS` (residual columns are altimetry minus DEM, in
    meters). The ``dh_*`` columns are each DEM's own surviving points; the
    ``*_shared_*`` columns are the ``n_shared`` points valid in every DEM
    (``shared_ids``), which is what the figure shows and the rows are sorted
    on. The ``median_ci_*`` / ``nmad_ci_*`` columns are 95 % bootstrap
    intervals on the shared-point statistics at the stage that is ranked
    (post-alignment when pc_align ran, ``bootstrap["stage"]``),
    ``nmad_vs_best_ci_*`` the interval on the DEM's NMAD minus the best
    DEM's on the same replicates (a DEM whose interval includes 0 is not
    separable from the best), and ``p_best`` the fraction of replicates on
    which the DEM has the lowest NMAD.

    ``altimetry`` maps each label to its :class:`Altimetry` object, for
    per-DEM figures such as ``mapview_plot_atl06sr_to_dem()`` or
    ``histogram_by_landcover()``; ``dh`` / ``dh_aligned`` map each label to
    the residual series scored, indexed by point id (aligned only when
    pc_align ran for it); ``blocks`` maps point id to the resampling block
    (beam track, or orbit) the bootstrap draws.
    """

    def __init__(
        self,
        directory,
        dems,
        parquet=None,
        altimetry_csv=None,
        key="all",
        reference=None,
        aoi="intersection",
        filter_out="water",
        n_sigma=3,
        n_bootstrap=1000,
        seed=0,
        title=None,
    ):
        self.directory = os.path.expanduser(directory)
        if isinstance(dems, dict):
            self.dems = {
                str(label): os.path.expanduser(fn) for label, fn in dems.items()
            }
        else:
            self.dems = {
                label: os.path.expanduser(fn)
                for label, fn in parse_dem_specs(list(dems)).items()
            }
        if not self.dems:
            raise ValueError("No DEMs given.")
        for label, fn in self.dems.items():
            if not os.path.exists(fn):
                raise ValueError(f"DEM file not found for '{label}': {fn}")

        bodies = {detect_planetary_body(fn) for fn in self.dems.values()}
        if len(bodies) > 1:
            raise ValueError(
                f"All DEMs must be on the same body; got {sorted(bodies)}."
            )
        self.body = bodies.pop()

        self.key = key
        if isinstance(parquet, str):
            parquet = {key: parquet}
        self.parquet = (
            {k: os.path.expanduser(v) for k, v in parquet.items()} if parquet else None
        )
        self.altimetry_csv = (
            os.path.expanduser(altimetry_csv) if altimetry_csv else None
        )
        if self.body == "earth":
            if self.parquet is None:
                raise ValueError(
                    "Earth DEMs are scored against ICESat-2: pass parquet= (the "
                    "atl06sr_<key>.parquet cache a report or "
                    "Altimetry.request_atl06sr_multi_processing(save_to_parquet=True) wrote)."
                )
            if self.key not in self.parquet:
                raise ValueError(
                    f"parquet has no entry for key='{self.key}': {sorted(self.parquet)}"
                )
            for k, v in self.parquet.items():
                if not os.path.exists(v):
                    raise ValueError(f"Parquet for '{k}' not found: {v}")
        else:
            if self.altimetry_csv is None:
                raise ValueError(
                    f"{self.body.capitalize()} DEMs are scored against LOLA/MOLA: "
                    "pass altimetry_csv= (see request_planetary_altimetry)."
                )
            if not os.path.exists(self.altimetry_csv):
                raise ValueError(f"Altimetry CSV not found: {self.altimetry_csv}")

        if reference is not None and reference not in self.dems:
            raise ValueError(
                f"reference '{reference}' is not one of the DEM labels: "
                f"{list(self.dems)}"
            )
        self.reference = reference
        self.filter_out = filter_out
        self.n_sigma = n_sigma
        if n_bootstrap < 0:
            raise ValueError("n_bootstrap must be 0 or a positive count.")
        self.n_bootstrap = int(n_bootstrap)
        self.seed = seed
        self.title = title

        self.aoi_bounds, self.aoi_crs = self._resolve_aoi(aoi)

        self.stats_df = None
        self.altimetry = {}
        self.dh = {}
        self.dh_aligned = {}
        self.blocks = None
        self.block_name = None
        self.shared_ids = None
        self.bootstrap = None
        self._pc_align_available = True

    # ------------------------------------------------------------------ #
    #  Setup                                                              #
    # ------------------------------------------------------------------ #

    @property
    def benchmark_directory(self):
        """Folder holding the per-label pc_align products and translated DEMs."""
        return os.path.join(self.directory, "dem_benchmark")

    def label_directory(self, label):
        """Working folder for one candidate."""
        return os.path.join(self.benchmark_directory, _safe_dirname(label))

    def _resolve_aoi(self, aoi):
        """Return (bounds, crs) for the coverage/IE window; (None, None) = own extent."""
        first_fn = next(iter(self.dems.values()))
        with rio.open(first_fn) as ds:
            crs0 = ds.crs
            b0 = tuple(ds.bounds)
        if aoi is None:
            return None, None
        if isinstance(aoi, (tuple, list)):
            if len(aoi) != 4:
                raise ValueError(
                    "aoi must be (left, bottom, right, top) or 'intersection'."
                )
            return tuple(float(v) for v in aoi), crs0
        if aoi != "intersection":
            raise ValueError(f"Unknown aoi: {aoi!r}")
        left, bottom, right, top = b0
        for fn in list(self.dems.values())[1:]:
            with rio.open(fn) as ds:
                b = tuple(ds.bounds)
                if ds.crs is not None and crs0 is not None and ds.crs != crs0:
                    b = transform_bounds(ds.crs, crs0, *b)
            left, bottom = max(left, b[0]), max(bottom, b[1])
            right, top = min(right, b[2]), min(top, b[3])
        if left >= right or bottom >= top:
            logger.warning(
                "The DEM footprints do not all overlap; coverage and "
                "triangulation-error statistics use each DEM's own extent."
            )
            return None, None
        return (left, bottom, right, top), crs0

    # ------------------------------------------------------------------ #
    #  Scoring                                                            #
    # ------------------------------------------------------------------ #

    def run(self, pc_align=True, minimum_points=500, max_displacement=None):
        """
        Score every DEM and return the stats table.

        Parameters
        ----------
        pc_align : bool, optional
            Also run ``pc_align --compute-translation-only`` per DEM and
            re-score against the translated copy. Default True. Existing
            pc_align outputs under the benchmark folder are reused, so a
            re-run makes no pc_align call and works offline.
        minimum_points : int, optional
            Skip pc_align for a DEM with fewer valid dh points, default 500
            (the report's threshold).
        max_displacement : float, optional
            ``pc_align --max-displacement`` in meters. Default None: 20 m on
            Earth, 500 m on the Moon/Mars, matching ``Alignment``.

        Returns
        -------
        pandas.DataFrame
            ``self.stats_df``: one row per DEM, columns :data:`STATS_COLUMNS`.
        """
        rows = []
        n = len(self.dems)
        for i, (label, dem_fn) in enumerate(self.dems.items(), start=1):
            print(f"\n=== [{i}/{n}] {label}: {dem_fn}")
            row = {"label": label, "dem_fn": dem_fn}
            row.update(self._coverage_stats(dem_fn))
            row.update(self._intersection_error_stats(dem_fn))
            row.update(
                self._altimetry_stats(
                    label,
                    dem_fn,
                    pc_align=pc_align and self._pc_align_available,
                    minimum_points=minimum_points,
                    max_displacement=max_displacement,
                )
            )
            row.update(self._reference_stats(label, dem_fn))
            rows.append(row)
        shared = self._shared_stats()
        for row in rows:
            row.update(shared[row["label"]])
        self.stats_df = pd.DataFrame(rows, columns=STATS_COLUMNS)
        return self.stats_df

    def _shared_stats(self):
        """
        Score every DEM on the points valid in all of them, and bootstrap.

        The shared set is the intersection of the point ids each DEM kept
        (finite dh after its outlier cut, and finite aligned dh where
        pc_align ran). A DEM with no points at all is left out of the
        intersection rather than emptying it. Returns ``{label: {column:
        value}}`` for the ``_SHARED_KEYS`` and ``_BOOTSTRAP_KEYS`` columns
        and fills ``shared_ids`` / ``bootstrap``.
        """
        empty = {k: np.nan for k in _SHARED_KEYS + _BOOTSTRAP_KEYS}
        out = {label: dict(empty) for label in self.dems}
        scored = [label for label, dh in self.dh.items() if len(dh)]
        if not scored:
            self.shared_ids = pd.Index([], dtype=int)
            return out
        shared = None
        for label in scored:
            ids = self.dh[label].index
            if label in self.dh_aligned:
                ids = ids.intersection(self.dh_aligned[label].index)
            shared = ids if shared is None else shared.intersection(ids)
        self.shared_ids = shared.sort_values()
        n_shared = len(self.shared_ids)
        print(
            f"\n{n_shared} altimetry points are valid in every DEM"
            + (
                f" ({len(scored)} of {len(self.dems)} DEMs have points)"
                if len(scored) < len(self.dems)
                else ""
            )
        )
        if n_shared == 0:
            return out

        before = {}
        aligned = {}
        for label in scored:
            before[label] = (
                self.dh[label].reindex(self.shared_ids).to_numpy(dtype=float)
            )
            out[label]["n_shared"] = n_shared
            out[label].update(_dh_stats(before[label], "dh_shared"))
            if label in self.dh_aligned:
                aligned[label] = (
                    self.dh_aligned[label]
                    .reindex(self.shared_ids)
                    .to_numpy(dtype=float)
                )
                out[label].update(_dh_stats(aligned[label], "dh_aligned_shared"))

        # Rank (and bootstrap) the aligned residuals when any DEM has them,
        # as _sorted_stats does; DEMs without stay out of the race.
        stage = "aligned" if aligned else "before"
        series = aligned if aligned else before
        labels = list(series)
        if self.n_bootstrap == 0 or not labels:
            self.bootstrap = None
            return out

        blocks = None
        if self.blocks is not None:
            blocks = self.blocks.reindex(self.shared_ids).to_numpy()
            if pd.isna(blocks).any():
                blocks = None
        if blocks is None and self.block_name is not None:
            logger.warning("Some shared points have no block id; resampling points.")
        values = np.vstack([series[label] for label in labels])
        result = paired_block_bootstrap(
            values, blocks=blocks, n_boot=self.n_bootstrap, seed=self.seed
        )
        point_nmad = np.array(
            [
                out[label][
                    f"dh_{'aligned_' if stage == 'aligned' else ''}shared_nmad_m"
                ]
                for label in labels
            ]
        )
        best = int(np.nanargmin(point_nmad)) if np.isfinite(point_nmad).any() else None
        med_lo, med_hi = np.nanpercentile(result["median"], [2.5, 97.5], axis=0)
        nmad_lo, nmad_hi = np.nanpercentile(result["nmad"], [2.5, 97.5], axis=0)
        ranks_first = np.zeros(len(labels))
        finite = np.isfinite(result["nmad"]).all(axis=1)
        if finite.any():
            winners = np.nanargmin(result["nmad"][finite], axis=1)
            ranks_first = np.bincount(winners, minlength=len(labels)) / finite.sum()
        if best is not None:
            diff = result["nmad"] - result["nmad"][:, [best]]
            diff_lo, diff_hi = np.nanpercentile(diff, [2.5, 97.5], axis=0)
        else:
            diff_lo = diff_hi = np.full(len(labels), np.nan)
        for i, label in enumerate(labels):
            out[label].update(
                {
                    "median_ci_low_m": float(med_lo[i]),
                    "median_ci_high_m": float(med_hi[i]),
                    "nmad_ci_low_m": float(nmad_lo[i]),
                    "nmad_ci_high_m": float(nmad_hi[i]),
                    "nmad_vs_best_ci_low_m": float(diff_lo[i]),
                    "nmad_vs_best_ci_high_m": float(diff_hi[i]),
                    "p_best": float(ranks_first[i]),
                }
            )
        self.bootstrap = {
            "stage": stage,
            "labels": labels,
            "best": labels[best] if best is not None else None,
            "n_replicates": self.n_bootstrap,
            "n_blocks": result["n_blocks"],
            "block": self.block_name if blocks is not None else "point",
            "median": result["median"],
            "nmad": result["nmad"],
        }
        n_sep = sum(
            1
            for label in labels
            if label != self.bootstrap["best"]
            and out[label]["nmad_vs_best_ci_low_m"] > 0
        )
        print(
            f"Paired bootstrap ({self.n_bootstrap} replicates of "
            f"{result['n_blocks']} {self.bootstrap['block']}s, {stage} pc_align): "
            f"best is '{self.bootstrap['best']}', P(first) = "
            f"{ranks_first[best]:.2f}; {len(labels) - 1 - n_sep} of {len(labels) - 1} "
            "other DEMs are not separable from it."
        )
        return out

    def _coverage_stats(self, dem_fn):
        with rio.open(dem_fn) as ds:
            gsd_x, gsd_y = abs(ds.transform.a), abs(ds.transform.e)
            win = _window_for_bounds(ds, self.aoi_bounds, self.aoi_crs)
            if win is None:
                logger.warning(f"{dem_fn} does not overlap the AOI; no coverage stats.")
                return {"gsd_m": gsd_x, "valid_pct": np.nan, "valid_area_km2": np.nan}
            a = _read_window(ds, win)
        valid_pct = 100.0 * a.count() / a.size if a.size else np.nan
        window_area_km2 = float(win.width) * float(win.height) * gsd_x * gsd_y / 1e6
        return {
            "gsd_m": gsd_x,
            "valid_pct": float(valid_pct),
            "valid_area_km2": float(valid_pct / 100.0 * window_area_km2),
        }

    def _intersection_error_stats(self, dem_fn):
        empty = {"ie_median_m": np.nan, "ie_nmad_m": np.nan}
        ie_fn = intersection_error_path(dem_fn)
        if ie_fn is None:
            return empty
        with rio.open(ie_fn) as ds:
            win = _window_for_bounds(ds, self.aoi_bounds, self.aoi_crs)
            if win is None:
                return empty
            a = _read_window(ds, win)
        values = a.compressed()
        if values.size == 0:
            return empty
        return {
            "ie_median_m": float(np.median(values)),
            "ie_nmad_m": float(nmad(values)),
        }

    def _altimetry_stats(
        self, label, dem_fn, pc_align, minimum_points, max_displacement
    ):
        label_dir = self.label_directory(label)
        os.makedirs(label_dir, exist_ok=True)
        alt = Altimetry(label_dir, dem_fn)
        self.altimetry[label] = alt

        if self.body == "earth":
            alt.load_atl06sr_from_parquet(self.parquet)
            self._tag_points(alt.atl06sr_processing_levels_filtered, self.key)
            if self.filter_out:
                alt.filter_esa_worldcover(filter_out=self.filter_out)
            alt.atl06sr_to_dem_dh(n_sigma=self.n_sigma)
            points = alt.atl06sr_processing_levels_filtered.get(self.key)
            dh_col, aligned_col = "icesat_minus_dem", "icesat_minus_aligned_dem"
        else:
            alt.load_planetary_csv(self.altimetry_csv)
            self._tag_planetary_points(alt)
            alt.planetary_to_dem_dh(n_sigma=self.n_sigma)
            points = alt.planetary_points
            dh_col, aligned_col = "altimetry_minus_dem", "altimetry_minus_aligned_dem"

        dh = self._series_by_point(points, dh_col)
        self.dh[label] = dh
        row = {"n_points": int(len(dh))}
        row.update(_dh_stats(dh, "dh"))
        row.update({k: np.nan for k in _ALIGNED_KEYS})
        print(
            f"  {len(dh)} altimetry points: median {row['dh_median_m']:+.2f} m, "
            f"NMAD {row['dh_nmad_m']:.2f} m"
        )

        if not pc_align:
            return row
        if len(dh) < minimum_points:
            print(
                f"  Skipping pc_align: {len(dh)} points is fewer than the "
                f"{minimum_points} required."
            )
            return row

        report, aligned_fn = self._align(alt, label_dir, dem_fn, max_displacement)
        if report is None or aligned_fn is None:
            return row
        alt.aligned_dem_fn = aligned_fn
        # Re-sample without re-filtering so the aligned residuals are computed
        # on exactly the points scored above (same as the report's success path).
        if self.body == "earth":
            alt.atl06sr_to_dem_dh(n_sigma=None)
            points = alt.atl06sr_processing_levels_filtered.get(self.key)
        else:
            alt.planetary_to_dem_dh(n_sigma=None)
            points = alt.planetary_points
        dh_aligned = self._series_by_point(points, aligned_col)
        if dh_aligned.empty:
            # The translated DEM sampled NaN everywhere: unreadable, or being
            # rewritten by another benchmark on the same label folder. Score
            # this DEM without alignment rather than let an empty aligned
            # series empty the shared point set for every DEM.
            logger.warning(
                f"No aligned residuals for '{label}' (is {aligned_fn} readable?); "
                "scoring it without alignment."
            )
            return row
        self.dh_aligned[label] = dh_aligned
        row.update(
            {
                "translation_m": report.get("translation_magnitude", np.nan),
                "north_shift_m": report.get("north_shift", np.nan),
                "east_shift_m": report.get("east_shift", np.nan),
                "down_shift_m": report.get("down_shift", np.nan),
            }
        )
        row.update(_dh_stats(dh_aligned, "dh_aligned"))
        print(
            f"  after pc_align (|t| = {row['translation_m']:.2f} m): median "
            f"{row['dh_aligned_median_m']:+.2f} m, NMAD {row['dh_aligned_nmad_m']:.2f} m"
        )
        return row

    def _tag_points(self, tables, key):
        """
        Give every point of ``tables[key]`` an id that survives the filters.

        The per-DEM ``Altimetry`` objects all replay the same parquet through
        the same ingest, so the row order is identical and a positional id
        assigned before any DEM-specific filtering identifies one point in
        every DEM's table. The tables keep a non-unique time index, which
        cannot serve. The first DEM tagged also records each point's
        resampling block (its beam track).
        """
        table = tables.get(key)
        if table is None or POINT_ID in table.columns:
            return
        tables[key] = table.assign(**{POINT_ID: np.arange(len(table))})
        if self.blocks is None:
            cols = {c.strip().lower(): c for c in table.columns}
            found = [cols[c] for c in ICESAT2_BLOCK_COLUMNS if c in cols]
            if len(found) < len(ICESAT2_BLOCK_COLUMNS):
                found = []
            self._record_blocks(tables[key], found, "beam track")

    def _tag_planetary_points(self, alt):
        """Planetary counterpart of :meth:`_tag_points`, on ``alt.planetary_points``.

        The CSV is read in the same order for every DEM and the outlier cut
        only drops rows, so a positional id serves here too. The block is the
        orbit when the CSV names one (MOLA PEDR does, LOLA RDR does not).
        """
        table = alt.planetary_points
        if table is None or POINT_ID in table.columns:
            return
        alt.planetary_points = table.assign(**{POINT_ID: np.arange(len(table))})
        if self.blocks is None:
            cols = {c.strip().lower(): c for c in table.columns}
            found = [cols[c] for c in PLANETARY_BLOCK_COLUMNS if c in cols][:1]
            self._record_blocks(alt.planetary_points, found, "orbit")

    def _record_blocks(self, table, columns, name):
        """Store the block id of every tagged point, from ``columns`` of ``table``."""
        if not columns:
            self.blocks = None
            self.block_name = None
            logger.warning(
                f"The altimetry points carry no {name} id; the bootstrap "
                "resamples points, which assumes they are independent and gives "
                "optimistic intervals."
            )
            return
        codes, _ = pd.factorize(pd.MultiIndex.from_frame(table[columns]))
        self.blocks = pd.Series(codes, index=table[POINT_ID].to_numpy())
        self.block_name = name

    @staticmethod
    def _series_by_point(points, column):
        """The finite values of ``points[column]`` as a Series indexed by point id."""
        if points is None or column not in points.columns:
            return pd.Series(dtype=float)
        series = pd.Series(
            points[column].to_numpy(dtype=float), index=points[POINT_ID].to_numpy()
        )
        return series[np.isfinite(series)]

    def _align(self, alt, label_dir, dem_fn, max_displacement):
        """Run (or reuse) pc_align for one DEM; return (report dict, translated DEM path)."""
        prefix_name = (
            f"pc_align_{self.key}" if self.body == "earth" else "pc_align_planetary"
        )
        output_prefix = f"pc_align/{prefix_name}"
        alignment = Alignment(label_dir, dem_fn)
        if glob_file(
            os.path.join(label_dir, "pc_align"),
            f"{prefix_name}-transform.txt",
            quiet=True,
        ):
            print(f"  Reusing pc_align output in {os.path.join(label_dir, 'pc_align')}")
        else:
            try:
                if self.body == "earth":
                    csv_fn = alt.to_csv_for_pc_align(key=self.key)
                    kwargs = {"atl06sr_csv": csv_fn, "output_prefix": output_prefix}
                    if max_displacement is not None:
                        kwargs["max_displacement"] = max_displacement
                    alignment.pc_align_dem_to_atl06sr(**kwargs)
                else:
                    csv_fn = alt.to_csv_for_pc_align_planetary()
                    kwargs = {"output_prefix": output_prefix}
                    if max_displacement is not None:
                        kwargs["max_displacement"] = max_displacement
                    alignment.pc_align_dem_to_planetary_csv(csv_fn, self.body, **kwargs)
            except FileNotFoundError as e:
                # pc_align itself is missing (ASP not on PATH): say so once and
                # score the remaining DEMs without alignment rather than failing.
                logger.warning(
                    f"pc_align could not be run ({e}); is the ASP bin directory on "
                    "PATH? Scoring without alignment."
                )
                self._pc_align_available = False
                return None, None
        report = alignment.pc_align_report(output_prefix=output_prefix)
        if not report:
            logger.warning(
                f"pc_align left no usable log under {os.path.join(label_dir, 'pc_align')}; "
                "scoring without alignment."
            )
            return None, None
        stem = os.path.splitext(os.path.basename(dem_fn))[0]
        aligned_fn = alignment.apply_dem_translation(
            output_prefix=output_prefix,
            output_fn=os.path.join(label_dir, f"{stem}_pc_align_translated.tif"),
        )
        return report, aligned_fn

    def _reference_stats(self, label, dem_fn):
        empty = {"vs_ref_median_m": np.nan, "vs_ref_nmad_m": np.nan}
        if self.reference is None:
            return empty
        if label == self.reference:
            return {"vs_ref_median_m": 0.0, "vs_ref_nmad_m": 0.0}
        try:
            # compute_difference returns (reference - dem) on the reference grid
            diff = Raster(dem_fn).compute_difference(self.dems[self.reference])
        except Exception as e:  # no overlap / CRS trouble: a NaN row, not a crash
            logger.warning(f"Could not difference '{label}' against the reference: {e}")
            return empty
        values = -np.ma.masked_invalid(diff).compressed()
        if values.size == 0:
            return empty
        return {
            "vs_ref_median_m": float(np.median(values)),
            "vs_ref_nmad_m": float(nmad(values)),
        }

    # ------------------------------------------------------------------ #
    #  Output                                                             #
    # ------------------------------------------------------------------ #

    def _require_stats(self):
        if self.stats_df is None or self.stats_df.empty:
            raise ValueError("No statistics yet: call run() first.")
        return self.stats_df

    def save_stats(self, csv_fn):
        """Write ``stats_df`` to ``csv_fn`` (creating its folder) and return the path."""
        df = self._require_stats()
        os.makedirs(os.path.dirname(os.path.abspath(csv_fn)), exist_ok=True)
        df.to_csv(csv_fn, index=False, float_format="%.4f")
        return csv_fn

    def _sorted_stats(self, sort):
        df = self._require_stats()
        if not sort:
            return df.reset_index(drop=True)
        by = (
            "dh_aligned_shared_nmad_m"
            if df["dh_aligned_shared_nmad_m"].notna().any()
            else "dh_shared_nmad_m"
        )
        return df.sort_values(by, na_position="last", kind="stable").reset_index(
            drop=True
        )

    def aoi_area_km2(self):
        """Area of the coverage/IE window in km² (None when each DEM uses its own extent)."""
        if self.aoi_bounds is None:
            return None
        left, bottom, right, top = self.aoi_bounds
        return (right - left) * (top - bottom) / 1e6

    def summary_plot(self, sort=True, save_dir=None, fig_fn=None, dpi=None):
        """
        One row per DEM, one panel per metric.

        Panels: coverage (% valid inside the AOI, area printed) and
        altimetry-minus-DEM median and NMAD on the points every DEM shares
        -- as dumbbells from before (open) to
        after (filled) pc_align when alignment ran, with the 95 % bootstrap
        interval as an error bar on the ranked marker. A translation-only
        pc_align leaves NMAD essentially unchanged by construction, so that
        panel's two markers coincide; the median panel is where the alignment
        shows. Rows are sorted best-first by post-alignment NMAD
        (pre-alignment when pc_align did not run) unless ``sort=False``; the
        best DEM's NMAD is labelled ``best`` and every DEM whose paired NMAD
        difference to it includes zero is labelled ``≈ best``.

        The IntersectionErr columns are not drawn. Across DEMs of different
        geometry the triangulation error ranks opposite to accuracy (a
        narrow pair's rays intersect precisely, at the wrong height), so a
        panel of it next to the residuals misleads; the columns stay in the
        table as a consistency check on a run.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = self._sorted_stats(sort)
        n = len(df)
        y = np.arange(n)[::-1]
        has_aligned = df["dh_aligned_shared_nmad_m"].notna().any()
        has_ci = df["nmad_ci_low_m"].notna().any()
        best = self.bootstrap["best"] if self.bootstrap else None

        panels = ["coverage", "median", "nmad"]
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(3.1 * len(panels) + 1.6, 0.36 * n + 2.2),
            sharey=True,
            dpi=220 if dpi is None else dpi,
        )
        axes = np.atleast_1d(axes)
        label_w = max(len(str(s)) for s in df["label"])
        fig.subplots_adjust(
            left=min(0.32, 0.06 + 0.011 * label_w),
            right=0.985,
            wspace=0.28,
            bottom=0.2,
        )

        def annotate(ax, x, texts, xmax=None):
            """Print ``texts`` right of the row markers and widen the axis so
            they fit, sizing the extension from the longest text and the
            panel's width on the page."""
            xs = np.asarray(x, dtype=float)
            finite = xs[np.isfinite(xs)]
            lo, hi = ax.get_xlim()
            if finite.size:
                lo = min(lo, finite.min())
                hi = max(hi, finite.max())
            if xmax is not None:
                hi = max(hi, xmax)
            span = hi - lo if hi > lo else 1.0
            width_in = ax.get_position().width * fig.get_figwidth()
            longest = max((len(t) for t in texts), default=0)
            text_frac = min(0.6, 0.055 * longest / max(width_in, 1e-6))
            extend = text_frac / (1.0 - text_frac) + 0.06
            ax.set_xlim(lo, hi + extend * span)
            for yi, xi, txt in zip(y, xs, texts):
                if np.isfinite(xi) and txt:
                    ax.text(
                        xi + 0.025 * span, yi, txt, va="center", ha="left", fontsize=6.5
                    )

        for ax, panel in zip(axes, panels):
            ax.grid(axis="x", color="0.9", lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
            if panel == "coverage":
                v = df["valid_pct"].to_numpy(dtype=float)
                ax.barh(y, np.nan_to_num(v), color="0.72", height=0.62)
                ax.set_xlim(0, 100)
                annotate(
                    ax,
                    v,
                    [
                        f"{p:.1f}% ({a:.1f} km²)" if np.isfinite(p) else ""
                        for p, a in zip(v, df["valid_area_km2"])
                    ],
                    xmax=100,
                )
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_xlabel("Valid inside AOI (%)", fontsize=8)
                ax.set_title("Coverage", fontsize=9)
            else:
                before = df[f"dh_shared_{panel}_m"].to_numpy(dtype=float)
                after = df[f"dh_aligned_shared_{panel}_m"].to_numpy(dtype=float)
                shown = (
                    np.where(np.isfinite(after), after, before)
                    if has_aligned
                    else before
                )
                if has_ci:
                    lo = df[f"{panel}_ci_low_m"].to_numpy(dtype=float)
                    hi = df[f"{panel}_ci_high_m"].to_numpy(dtype=float)
                    ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(shown)
                    ax.errorbar(
                        shown[ok],
                        y[ok],
                        xerr=[shown[ok] - lo[ok], hi[ok] - shown[ok]],
                        fmt="none",
                        ecolor="tab:blue" if has_aligned else "0.25",
                        elinewidth=1.0,
                        capsize=2,
                        zorder=1,
                    )
                if has_aligned:
                    ax.hlines(y, before, after, color="0.6", lw=1.4, zorder=1)
                    ax.scatter(
                        after,
                        y,
                        s=26,
                        color="tab:blue",
                        zorder=2,
                        label="after pc_align",
                    )
                ax.scatter(
                    before,
                    y,
                    s=26,
                    facecolors="none",
                    edgecolors="0.25",
                    linewidths=1.0,
                    zorder=3,
                    label="before pc_align" if has_aligned else None,
                )
                if panel == "median":
                    ax.axvline(0, color="k", lw=0.6, zorder=0)
                else:
                    # NMAD is a spread: anchor the axis at zero so the row-to-row
                    # differences are shown at their true proportion.
                    ax.set_xlim(0, ax.get_xlim()[1])
                fmt = "{:+.2f}" if panel == "median" else "{:.2f}"
                texts = []
                for i, v in enumerate(shown):
                    if not np.isfinite(v):
                        texts.append("")
                        continue
                    txt = fmt.format(v)
                    if has_ci and np.isfinite(lo[i]) and np.isfinite(hi[i]):
                        txt += f" [{fmt.format(lo[i])}, {fmt.format(hi[i])}]"
                    if panel == "nmad" and best is not None:
                        if df.loc[i, "label"] == best:
                            txt += "  best"
                        elif not df.loc[i, "nmad_vs_best_ci_low_m"] > 0:
                            txt += "  ≈ best"
                    texts.append(txt)
                right_edge = np.fmax(
                    np.nan_to_num(before, nan=-np.inf),
                    np.nan_to_num(shown, nan=-np.inf),
                )
                if has_ci:
                    right_edge = np.fmax(right_edge, np.nan_to_num(hi, nan=-np.inf))
                annotate(ax, right_edge, texts)
                what = "median" if panel == "median" else "NMAD"
                ax.set_xlabel(f"Altimetry − DEM, {what} (m)", fontsize=8)
                ax.set_title(f"dh {what}", fontsize=9)
                if panel == "median" and has_aligned:
                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.2),
                        ncol=2,
                        fontsize=6.5,
                        frameon=False,
                        handletextpad=0.3,
                        columnspacing=1.0,
                    )

        axes[0].set_yticks(y)
        axes[0].set_yticklabels(df["label"], fontsize=7.5)
        axes[0].set_ylim(-0.7, n - 0.3)

        n_pts = df["n_points"].to_numpy(dtype=float)
        parts = [
            (
                "ICESat-2 ATL06-SR"
                if self.body == "earth"
                else f"{self.body.capitalize()} altimetry"
            )
        ]
        n_shared = df["n_shared"].dropna()
        if len(n_shared):
            own = ""
            if np.isfinite(n_pts).any():
                own = f" (of {int(np.nanmin(n_pts))}–{int(np.nanmax(n_pts))} per DEM)"
            parts.append(f"n = {int(n_shared.iloc[0])} points valid in every DEM{own}")
        elif np.isfinite(n_pts).any():
            parts.append(
                f"n = {int(np.nanmin(n_pts))}–{int(np.nanmax(n_pts))} points per DEM"
            )
        area = self.aoi_area_km2()
        if area is not None:
            parts.append(f"common AOI {area:.1f} km²")
        lines = [", ".join(parts)]
        if has_ci and self.bootstrap:
            b = self.bootstrap
            lines.append(
                f"error bars: 95 % interval from {b['n_replicates']} replicates "
                f"resampling {b['n_blocks']} {b['block']}s"
            )
            lines.append(
                "≈ best: the paired NMAD difference to the best DEM includes 0"
            )
        if self.title:
            lines.insert(0, self.title)
        # Reserve the top margin for the title lines rather than a fixed
        # fraction, so a tall figure (many DEMs) has no gap under them and a
        # short one is not cramped: ~0.2 in per line plus a little air.
        fig.subplots_adjust(top=1 - (0.25 + 0.2 * len(lines)) / fig.get_figheight())
        fig.suptitle("\n".join(lines), fontsize=10 if self.title else 9)
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn, dpi=dpi)
        return fig

    def histogram_plot(
        self,
        aligned="auto",
        bins=100,
        xlim=None,
        sort=True,
        save_dir=None,
        fig_fn=None,
        dpi=None,
    ):
        """
        Overlaid residual histograms, one outline per DEM, on the points
        every DEM shares (each DEM's own points when there are none).

        Parameters
        ----------
        aligned : {"auto", True, False}, optional
            Plot post-alignment residuals when every DEM has them
            (``"auto"``, default), always (True, DEMs without them fall back
            to pre-alignment), or never (False).
        bins : int, optional
            Histogram bins, default 100.
        xlim : float, optional
            Half-width of the plotted range in meters; default 3× the largest
            NMAD among the DEMs. Residuals beyond it are not drawn (the legend
            counts all of them).
        sort : bool, optional
            Order the legend best-first as in :meth:`summary_plot`, default True.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = self._sorted_stats(sort)
        if aligned == "auto":
            use_aligned = bool(self.dh_aligned) and all(
                label in self.dh_aligned for label in df["label"]
            )
        else:
            use_aligned = bool(aligned)
        shared = self.shared_ids if self.shared_ids is not None else pd.Index([])
        series = {}
        for label in df["label"]:
            dh = self.dh_aligned.get(label) if use_aligned else None
            if dh is None:
                dh = self.dh.get(label, pd.Series(dtype=float))
            if len(shared) and len(dh):
                dh = dh.reindex(shared)
            series[label] = np.asarray(dh, dtype=float)
        if xlim is None:
            nmads = [nmad(v) for v in series.values() if np.isfinite(v).sum() > 1]
            xlim = 3.0 * max(nmads) if nmads else 1.0
            xlim = max(xlim, 0.1)

        fig, ax = plt.subplots(1, 1, figsize=(7, 4.2), dpi=220 if dpi is None else dpi)
        for label, v in series.items():
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            med, spread = np.median(v), nmad(v)
            ax.hist(
                v,
                bins=bins,
                range=(-xlim, xlim),
                histtype="step",
                density=True,
                lw=1.2,
                label=f"{label}: median {med:+.2f} m, NMAD {spread:.2f} m (n={v.size})",
            )
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(-xlim, xlim)
        state = "after pc_align" if use_aligned else "before pc_align"
        if len(shared):
            state += ", points valid in every DEM"
        ax.set_xlabel(f"Altimetry − DEM (m), {state}")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6.5, frameon=False)
        if self.title:
            ax.set_title(self.title, fontsize=10)
        fig.tight_layout()
        if save_dir and fig_fn:
            save_figure(fig, save_dir, fig_fn, dpi=dpi)
        return fig
