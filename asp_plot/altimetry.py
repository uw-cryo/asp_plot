"""Altimetry analysis coordinator.

:class:`Altimetry` is a thin coordinator that composes three single-concern
objects and exposes their behaviour under one notebook-friendly API:

- :class:`asp_plot.icesat2_source.Icesat2Source` — ICESat-2 ATL06-SR fetch,
  caching, WorldCover sampling, temporal/outlier filtering, DEM differencing,
  and track/segment selection.
- :class:`asp_plot.planetary_source.PlanetarySource` — LOLA/MOLA loading and
  DEM differencing, via the per-body :class:`~asp_plot.planetary_source.LolaSource`
  / :class:`~asp_plot.planetary_source.MolaSource` subclasses selected from the
  DEM's body at construction.
- :class:`asp_plot.altimetry_plots.AltimetryPlotter` — all figure rendering,
  operating on prepared dataframes.

The coordinator owns the cross-cutting state that describes the DEM under
analysis (``directory``, ``dem_fn``, ``aligned_dem_fn``) and the pc_align
orchestration + keep/discard decision shared by the Earth and planetary
paths (#127). The source/plotter objects read that state through a back
reference. ``Alignment`` and ``Raster`` are imported at module scope because
the alignment methods reference them directly.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from asp_plot.alignment import Alignment
from asp_plot.altimetry_plots import AltimetryPlotter
from asp_plot.bodies import BODIES
from asp_plot.icesat2_source import ICESAT2_MISSION_START, Icesat2Source  # noqa: F401
from asp_plot.planetary_source import (  # noqa: F401
    GDS_BASE_URL,
    LolaSource,
    MolaSource,
    PlanetarySource,
    gds_query_async,
)
from asp_plot.utils import Raster, detect_planetary_body, glob_file

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Outcome of :meth:`Altimetry.align_and_evaluate`.

    Plain dataclass — importing this or the method does not pull in any
    report/fpdf dependencies, so it is safe to use from notebooks.

    Attributes
    ----------
    status : str
        One of:
          - ``"insufficient_points"``: not enough ATL06-SR points for
            pc_align to run (the aligned DEM is removed if one was written).
          - ``"no_improvement"``: pc_align ran but p50 did not improve
            toward 0 by more than the ``improvement_threshold_pct``; the
            aligned DEM has been removed.
          - ``"success"``: p50 improved by more than the threshold; the
            aligned DEM is retained and ``Altimetry.aligned_dem_fn`` points
            to it.
    alignment_report_df : pandas.DataFrame
        The alignment report table produced by
        :meth:`Altimetry.alignment_report`. Empty for ``insufficient_points``.
    aligned_dem_fn : str or None
        Path to the aligned DEM on success; None otherwise (the file is
        cleaned up on non-success branches).
    improvement_pct : float or None
        ``(p50_beg - p50_end) / p50_beg * 100`` when computable, else None.
    message : str
        Short human-readable summary suitable for a status line in the
        PDF report.
    parameters_used : dict
        The kwargs used for the alignment attempt (e.g. processing_level,
        minimum_points, improvement_threshold_pct). Echoed into the report.
    """

    status: str
    alignment_report_df: Optional[pd.DataFrame] = None
    aligned_dem_fn: Optional[str] = None
    improvement_pct: Optional[float] = None
    message: str = ""
    parameters_used: dict = field(default_factory=dict)


class Altimetry:
    """
    Process and analyze ICESat-2 / planetary altimetry against ASP DEMs.

    Coordinates the ICESat-2 (:class:`Icesat2Source`) and planetary
    (:class:`PlanetarySource`, i.e. :class:`LolaSource` / :class:`MolaSource`)
    data sources and the plotting layer (:class:`AltimetryPlotter`), exposing
    them under a single API. It can
    request and filter altimetry data, align a DEM to it, and visualize the
    results.

    Attributes
    ----------
    directory : str
        Root directory for outputs and analysis
    dem_fn : str
        Path to the DEM file to analyze
    aligned_dem_fn : str or None
        Path to the aligned DEM file if available
    atl06sr_processing_levels : dict
        Dictionary of ATL06-SR data for different processing levels
    atl06sr_processing_levels_filtered : dict
        Dictionary of filtered ATL06-SR data for different processing levels
    planetary_points : geopandas.GeoDataFrame or None
        Loaded LOLA/MOLA altimetry points (planetary DEMs)
    alignment_report_df : pandas.DataFrame or None
        DataFrame containing alignment reports when available

    Examples
    --------
    >>> altimetry = Altimetry('/path/to/directory', '/path/to/dem.tif')
    >>> altimetry.request_atl06sr_multi_processing(save_to_parquet=True)
    >>> altimetry.filter_esa_worldcover(filter_out="water")
    >>> altimetry.alignment_report()
    >>> altimetry.mapview_plot_atl06sr_to_dem()
    """

    def __init__(
        self,
        directory,
        dem_fn,
        aligned_dem_fn=None,
        atl06sr_processing_levels=None,
        atl06sr_processing_levels_filtered=None,
        **kwargs,
    ):
        """
        Initialize the Altimetry object.

        Parameters
        ----------
        directory : str
            Root directory for outputs and analysis
        dem_fn : str
            Path to the DEM file to analyze
        aligned_dem_fn : str, optional
            Path to an already aligned DEM file, default is None
        atl06sr_processing_levels : dict, optional
            Pre-loaded ATL06-SR data for different processing levels
        atl06sr_processing_levels_filtered : dict, optional
            Pre-loaded filtered ATL06-SR data
        **kwargs : dict, optional
            Additional keyword arguments for future extensions

        Raises
        ------
        ValueError
            If the DEM file or aligned DEM file (if provided) does not exist
        """
        self.directory = os.path.expanduser(directory)

        dem_fn = os.path.expanduser(dem_fn)
        if not os.path.exists(dem_fn):
            raise ValueError(f"DEM file not found: {dem_fn}")
        self.dem_fn = dem_fn

        if aligned_dem_fn is not None:
            aligned_dem_fn = os.path.expanduser(aligned_dem_fn)
            if not os.path.exists(aligned_dem_fn):
                raise ValueError(f"Aligned DEM file not found: {aligned_dem_fn}")
        self.aligned_dem_fn = aligned_dem_fn

        # Single-concern components. Each holds a back reference to this
        # coordinator so the DEM/directory/aligned-DEM live in exactly one
        # place. The plotter receives prepared dataframes per call.
        self.icesat2 = Icesat2Source(
            self,
            atl06sr_processing_levels=atl06sr_processing_levels,
            atl06sr_processing_levels_filtered=atl06sr_processing_levels_filtered,
        )
        # Pick the planetary source from the DEM's body so LOLA/MOLA loading is
        # dispatched once, at construction. Earth DEMs get the body-agnostic
        # base (its load_planetary_csv raises, redirecting to ICESat-2).
        planetary_cls = {"moon": LolaSource, "mars": MolaSource}.get(
            detect_planetary_body(dem_fn), PlanetarySource
        )
        self.planetary = planetary_cls(self)
        self.plotter = AltimetryPlotter(self)

    # ------------------------------------------------------------------ #
    #  Back-compat state access (delegates to the composed sources)       #
    # ------------------------------------------------------------------ #

    @property
    def atl06sr_processing_levels(self):
        return self.icesat2.atl06sr_processing_levels

    @atl06sr_processing_levels.setter
    def atl06sr_processing_levels(self, value):
        self.icesat2.atl06sr_processing_levels = value

    @property
    def atl06sr_processing_levels_filtered(self):
        return self.icesat2.atl06sr_processing_levels_filtered

    @atl06sr_processing_levels_filtered.setter
    def atl06sr_processing_levels_filtered(self, value):
        self.icesat2.atl06sr_processing_levels_filtered = value

    @property
    def planetary_points(self):
        return self.planetary.planetary_points

    @planetary_points.setter
    def planetary_points(self, value):
        self.planetary.planetary_points = value

    # ------------------------------------------------------------------ #
    #  ICESat-2 source delegation                                         #
    # ------------------------------------------------------------------ #

    def request_atl06sr_multi_processing(self, *args, **kwargs):
        return self.icesat2.request_atl06sr_multi_processing(*args, **kwargs)

    def load_atl06sr_from_parquet(self, *args, **kwargs):
        return self.icesat2.load_atl06sr_from_parquet(*args, **kwargs)

    def sample_esa_worldcover(self, *args, **kwargs):
        return self.icesat2.sample_esa_worldcover(*args, **kwargs)

    def filter_esa_worldcover(self, *args, **kwargs):
        return self.icesat2.filter_esa_worldcover(*args, **kwargs)

    def filter_outliers(self, *args, **kwargs):
        return self.icesat2.filter_outliers(*args, **kwargs)

    def predefined_temporal_filter_atl06sr(self, *args, **kwargs):
        return self.icesat2.predefined_temporal_filter_atl06sr(*args, **kwargs)

    def generic_temporal_filter_atl06sr(self, *args, **kwargs):
        return self.icesat2.generic_temporal_filter_atl06sr(*args, **kwargs)

    def to_csv_for_pc_align(self, *args, **kwargs):
        return self.icesat2.to_csv_for_pc_align(*args, **kwargs)

    def atl06sr_to_dem_dh(self, n_sigma=3):
        return self.icesat2.atl06sr_to_dem_dh(n_sigma=n_sigma)

    def get_altimetry_selections(self, *args, **kwargs):
        return self.icesat2.get_altimetry_selections(*args, **kwargs)

    # ------------------------------------------------------------------ #
    #  Planetary source delegation                                        #
    # ------------------------------------------------------------------ #

    def load_planetary_csv(self, *args, **kwargs):
        return self.planetary.load_planetary_csv(*args, **kwargs)

    def planetary_to_dem_dh(self, *args, **kwargs):
        return self.planetary.planetary_to_dem_dh(*args, **kwargs)

    def to_csv_for_pc_align_planetary(self, filename_prefix="planetary_for_pc_align"):
        return self.planetary.to_csv_for_pc_align_planetary(
            filename_prefix=filename_prefix
        )

    # ------------------------------------------------------------------ #
    #  ICESat-2 plotting (prepare dataframes, then delegate to plotter)   #
    # ------------------------------------------------------------------ #

    def plot_atl06sr_time_stamps(self, key="all", **kwargs):
        return self.plotter.plot_atl06sr_time_stamps(
            self.icesat2.atl06sr_processing_levels_filtered, key=key, **kwargs
        )

    def plot_atl06sr(self, key="all", **kwargs):
        atl06sr = self.icesat2.atl06sr_processing_levels_filtered[key]
        return self.plotter.plot_atl06sr(atl06sr, key=key, **kwargs)

    def mapview_plot_atl06sr_to_dem(
        self,
        key="all",
        clim=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
        map_crs=None,
        **ctx_kwargs,
    ):
        if plot_aligned and not self.aligned_dem_fn:
            print("\nAligned DEM not found.\n")
            return
        column_name = "icesat_minus_aligned_dem" if plot_aligned else "icesat_minus_dem"
        atl06sr = self.icesat2.atl06sr_processing_levels_filtered[key]
        if column_name not in atl06sr.columns:
            print(
                f"\n{column_name} not found in ATL06 dataframe: {key}. "
                "Running differencing first.\n"
            )
            self.atl06sr_to_dem_dh()
            atl06sr = self.icesat2.atl06sr_processing_levels_filtered[key]
        return self.plotter.mapview_plot_atl06sr_to_dem(
            atl06sr,
            key=key,
            clim=clim,
            plot_aligned=plot_aligned,
            save_dir=save_dir,
            fig_fn=fig_fn,
            map_crs=map_crs,
            **ctx_kwargs,
        )

    def histogram(
        self,
        key="all",
        title="Histogram",
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        atl06sr = self.icesat2.atl06sr_processing_levels_filtered[key]
        if "icesat_minus_dem" not in atl06sr.columns:
            print(
                f"\n'icesat_minus_dem' not found in ATL06 dataframe: {key}. "
                "Running differencing first.\n"
            )
            self.atl06sr_to_dem_dh()
            atl06sr = self.icesat2.atl06sr_processing_levels_filtered[key]
        return self.plotter.histogram(
            atl06sr,
            key=key,
            title=title,
            plot_aligned=plot_aligned,
            save_dir=save_dir,
            fig_fn=fig_fn,
        )

    def histogram_by_landcover(
        self,
        key="all",
        top_n=4,
        title="ICESat-2 ATL06-SR vs DEM",
        xlim=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        atl06sr = self.icesat2.atl06sr_processing_levels_filtered[key]
        if "icesat_minus_dem" not in atl06sr.columns:
            self.atl06sr_to_dem_dh()
            atl06sr = self.icesat2.atl06sr_processing_levels_filtered[key]
        return self.plotter.histogram_by_landcover(
            atl06sr,
            key=key,
            top_n=top_n,
            title=title,
            xlim=xlim,
            plot_aligned=plot_aligned,
            save_dir=save_dir,
            fig_fn=fig_fn,
        )

    def plot_atl06sr_dem_profile(
        self,
        key="all",
        rgt=None,
        cycle=None,
        spot=None,
        segments=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        resolved = self.icesat2._resolve_best_track(key, rgt, cycle, spot)
        if resolved is None:
            return
        track = resolved[0]
        seg_info = self.icesat2._find_best_worst_segments(
            track, segment_override=segments
        )
        return self.plotter.plot_atl06sr_dem_profile(
            resolved,
            seg_info,
            plot_aligned=plot_aligned,
            save_dir=save_dir,
            fig_fn=fig_fn,
        )

    def plot_best_worst_segments(
        self,
        key="all",
        rgt=None,
        cycle=None,
        spot=None,
        segments=None,
        plot_aligned=False,
        save_dir=None,
        fig_fn=None,
    ):
        resolved = self.icesat2._resolve_best_track(key, rgt, cycle, spot)
        if resolved is None:
            return
        track = resolved[0]
        seg_info = self.icesat2._find_best_worst_segments(
            track, segment_override=segments
        )
        if seg_info is None:
            logger.warning(
                "\nTrack too short or insufficient data for segment selection.\n"
            )
            return
        return self.plotter.plot_best_worst_segments(
            resolved,
            seg_info,
            plot_aligned=plot_aligned,
            save_dir=save_dir,
            fig_fn=fig_fn,
        )

    # ------------------------------------------------------------------ #
    #  Planetary plotting (prepare dataframes, then delegate to plotter)  #
    # ------------------------------------------------------------------ #

    def mapview_plot_planetary_to_dem(
        self,
        clim=None,
        save_dir=None,
        fig_fn=None,
        title=None,
        plot_aligned=False,
    ):
        points = self.planetary.planetary_points
        if points is None or points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return
        if "altimetry_minus_dem" not in points.columns:
            self.planetary_to_dem_dh()
        return self.plotter.mapview_plot_planetary_to_dem(
            self.planetary.planetary_points,
            clim=clim,
            save_dir=save_dir,
            fig_fn=fig_fn,
            title=title,
            plot_aligned=plot_aligned,
        )

    def histogram_planetary_to_dem(
        self,
        save_dir=None,
        fig_fn=None,
        title=None,
        plot_aligned=False,
    ):
        points = self.planetary.planetary_points
        if points is None or points.empty:
            logger.warning("No planetary altimetry points loaded.")
            return
        if "altimetry_minus_dem" not in points.columns:
            self.planetary_to_dem_dh()
        return self.plotter.histogram_planetary_to_dem(
            self.planetary.planetary_points,
            save_dir=save_dir,
            fig_fn=fig_fn,
            title=title,
            plot_aligned=plot_aligned,
        )

    # ------------------------------------------------------------------ #
    #  pc_align orchestration + shared keep/discard decision (#127)       #
    # ------------------------------------------------------------------ #

    def alignment_report(
        self,
        processing_level="ground",
        minimum_points=500,
        agreement_threshold=0.25,
        write_out_aligned_dem=False,
        min_translation_threshold=0.1,
        key_for_aligned_dem=None,
    ):
        """
        Generate alignment reports and optionally align the DEM.

        Runs pc_align between the DEM and filtered ATL06-SR data for all
        temporal variations of a given processing level, generates reports
        of the alignment results, and optionally creates an aligned DEM.

        Parameters
        ----------
        processing_level : str, optional
            Base processing level to use, default is "ground"
        minimum_points : int, optional
            Minimum number of points required for alignment, default is 500
        agreement_threshold : float, optional
            Threshold for agreement between different temporal alignments,
            as a fraction of the mean shift, default is 0.25
        write_out_aligned_dem : bool, optional
            Whether to create an aligned DEM, default is False
        min_translation_threshold : float, optional
            Minimum translation magnitude as a fraction of the DEM GSD
            to warrant creating an aligned DEM, default is 0.1
        key_for_aligned_dem : str, optional
            Which temporal filter key to use for alignment if
            write_out_aligned_dem is True. Default is None, which
            uses the ``processing_level`` value.

        Returns
        -------
        None
            Sets self.alignment_report_df and optionally self.aligned_dem_fn

        Notes
        -----
        This method performs both the pc_align operations and analysis of the
        alignment results. It checks for consistency across temporal filters
        and only creates an aligned DEM if the translation is significant enough
        and consistent across temporal windows.
        """
        if key_for_aligned_dem is None:
            key_for_aligned_dem = processing_level

        filtered_keys = [
            key
            for key in self.atl06sr_processing_levels_filtered.keys()
            if key.startswith(processing_level)
        ]

        alignment = Alignment(self.directory, self.dem_fn)

        for key in filtered_keys:
            atl06sr = self.atl06sr_processing_levels_filtered[key]
            if atl06sr.shape[0] < minimum_points:
                print(
                    f"\n{key} has {atl06sr.shape[0]} points, which is less than the suggested {minimum_points} points. Skipping alignment.\n"
                )
                continue

            if not glob_file(
                os.path.join(self.directory, "pc_align"), f"*{key}*transform.txt"
            ):
                csv_fn = self.to_csv_for_pc_align(key=key)

                print(f"\nAligning {key} to DEM with pc_align\n")

                alignment.pc_align_dem_to_atl06sr(
                    atl06sr_csv=csv_fn,
                    output_prefix=f"pc_align/pc_align_{key}",
                )

        report_data = []
        for key in filtered_keys:
            report = alignment.pc_align_report(output_prefix=f"pc_align/pc_align_{key}")
            if report:
                report_data.append({"key": key} | report)
        alignment_report_df = pd.DataFrame(report_data)

        if alignment_report_df.empty:
            logger.warning(
                f"\nNo alignment results for processing_level='{processing_level}'. "
                "Check that the requested processing level was included in "
                "request_atl06sr_multi_processing().\n"
            )
            self.alignment_report_df = alignment_report_df
            return

        gsd = Raster(self.dem_fn).get_gsd()
        if (
            alignment_report_df["translation_magnitude"].mean()
            < min_translation_threshold * gsd
        ):
            write_out_aligned_dem = False
            print(
                f"\nTranslation magnitude is less than {min_translation_threshold*100}% of the DEM GSD. Skipping writing out aligned DEM.\n"
            )

        if write_out_aligned_dem:
            # Calculate ranges and mean for each shift component (North-East-Down)
            north_range = (
                alignment_report_df["north_shift"].max()
                - alignment_report_df["north_shift"].min()
            )
            east_range = (
                alignment_report_df["east_shift"].max()
                - alignment_report_df["east_shift"].min()
            )
            down_range = (
                alignment_report_df["down_shift"].max()
                - alignment_report_df["down_shift"].min()
            )
            north_mean = alignment_report_df["north_shift"].mean()
            east_mean = alignment_report_df["east_shift"].mean()
            down_mean = alignment_report_df["down_shift"].mean()

            # Check if range is more than X% of mean for any component
            if (
                north_range > abs(north_mean * agreement_threshold)
                or east_range > abs(east_mean * agreement_threshold)
                or down_range > abs(down_mean * agreement_threshold)
            ):
                print(
                    f"\nWarning: Translation components vary by more than {agreement_threshold*100}% across temporal filters. The translation applied to the aligned DEM may be inaccurate.\n"
                )
                print(
                    f"North shift range: {north_range:.3f} m (mean: {north_mean:.3f} m)"
                )
                print(f"East shift range: {east_range:.3f} m (mean: {east_mean:.3f} m)")
                print(f"Down shift range: {down_range:.3f} m (mean: {down_mean:.3f} m)")

            aligned = alignment.apply_dem_translation(
                output_prefix=f"pc_align/pc_align_{key_for_aligned_dem}",
            )
            if aligned is not None:
                self.aligned_dem_fn = aligned
                print(
                    f"\nWrote out {key_for_aligned_dem} aligned DEM to {self.aligned_dem_fn}\n"
                )
            else:
                logger.warning(
                    f"Could not apply DEM translation for key '{key_for_aligned_dem}' "
                    "(pc_align log not found)."
                )

        self.alignment_report_df = alignment_report_df

    def align_and_evaluate(
        self,
        processing_level="all",
        improvement_threshold_pct=5.0,
        min_translation_threshold=0.1,
        minimum_points=500,
        agreement_threshold=0.25,
    ):
        """
        Run pc_align against ICESat-2 and evaluate whether to keep the result.

        Wraps :meth:`alignment_report` with a decision step so the aligned
        DEM is only retained when it represents a meaningful improvement.
        Returns an :class:`AlignmentResult`; notebook callers can inspect
        ``result.status`` to decide what to display.

        Decision logic:

        1. If the alignment report is empty (fewer than ``minimum_points``
           ICESat-2 points, or the pc_align log is missing), delete any
           aligned DEM file and return ``status="insufficient_points"``.
        2. Otherwise compute
           ``improvement_pct = (p50_beg - p50_end) / p50_beg * 100``. If
           ``p50_end >= p50_beg`` or
           ``improvement_pct <= improvement_threshold_pct``, delete the
           aligned DEM, clear ``self.aligned_dem_fn``, and return
           ``status="no_improvement"``.
        3. Otherwise re-run :meth:`atl06sr_to_dem_dh` so the
           ``icesat_minus_aligned_dem`` column is populated, and return
           ``status="success"`` with the aligned DEM retained.

        Parameters
        ----------
        processing_level : str, optional
            ATL06-SR processing level key to align against. Default "all".
        improvement_threshold_pct : float, optional
            Minimum required ``(p50_beg - p50_end) / p50_beg * 100`` for the
            aligned DEM to be kept. Default 5.0.
        min_translation_threshold : float, optional
            Forwarded to :meth:`alignment_report`. Default 0.1.
        minimum_points : int, optional
            Forwarded to :meth:`alignment_report`. Default 500.
        agreement_threshold : float, optional
            Forwarded to :meth:`alignment_report`. Default 0.25.

        Returns
        -------
        AlignmentResult
        """
        parameters_used = {
            "processing_level": processing_level,
            "minimum_points": minimum_points,
            "agreement_threshold": agreement_threshold,
            "min_translation_threshold": min_translation_threshold,
            "improvement_threshold_pct": improvement_threshold_pct,
        }

        self.alignment_report(
            processing_level=processing_level,
            key_for_aligned_dem=processing_level,
            minimum_points=minimum_points,
            agreement_threshold=agreement_threshold,
            min_translation_threshold=min_translation_threshold,
            write_out_aligned_dem=True,
        )
        df = getattr(self, "alignment_report_df", None)

        if df is None or df.empty:
            self._remove_aligned_dem_if_present()
            return AlignmentResult(
                status="insufficient_points",
                alignment_report_df=df if df is not None else pd.DataFrame(),
                aligned_dem_fn=None,
                improvement_pct=None,
                message=(
                    f"Alignment skipped: fewer than {minimum_points} "
                    f"ATL06-SR points available for processing_level="
                    f"'{processing_level}', or pc_align did not produce a "
                    "usable log."
                ),
                parameters_used=parameters_used,
            )

        row = df.iloc[0]
        p50_beg = float(row.get("p50_beg", float("nan")))
        p50_end = float(row.get("p50_end", float("nan")))
        improvement_pct = self._improvement_pct(p50_beg, p50_end)

        # alignment_report may decline to write the aligned DEM if the
        # translation magnitude is under min_translation_threshold × GSD
        # (self.aligned_dem_fn stays None in that case). Without a written
        # aligned DEM we cannot render the success-path plots, so treat
        # that as no_improvement even if p50 happened to drop > threshold.
        translation_too_small = self.aligned_dem_fn is None

        improvement_repr = (
            f"{improvement_pct:.1f}%" if improvement_pct is not None else "n/a"
        )
        if (
            translation_too_small
            and improvement_pct is not None
            and (p50_end < p50_beg and improvement_pct > improvement_threshold_pct)
        ):
            reason = (
                f"Translation magnitude is below {min_translation_threshold*100:.0f}% "
                "of the DEM GSD, so no aligned DEM was written despite a "
                f"{improvement_repr} p50 reduction."
            )
        else:
            reason = (
                f"p50 {p50_beg:.2f} m -> {p50_end:.2f} m, "
                f"{improvement_repr} <= {improvement_threshold_pct:.1f}% "
                "threshold. Aligned DEM removed."
            )

        result = self._evaluate_improvement(
            df=df,
            p50_beg=p50_beg,
            p50_end=p50_end,
            improvement_pct=improvement_pct,
            improvement_threshold_pct=improvement_threshold_pct,
            translation_too_small=translation_too_small,
            no_improvement_reason=reason,
            parameters_used=parameters_used,
        )
        if result is not None:
            return result

        # Populate icesat_minus_aligned_dem without re-running the 3σ
        # outlier filter (the unaligned column is already 3σ-clean from the
        # initial atl06sr_to_dem_dh call, and we do not want the aligned-DEM
        # plots to operate on a different sample than the unaligned ones).
        # This does not re-request ICESat-2 data; it only interpolates DEM
        # heights at the existing ATL06-SR point locations.
        self.atl06sr_to_dem_dh(n_sigma=None)
        return self._success_result(
            df, p50_beg, p50_end, improvement_pct, parameters_used
        )

    def align_and_evaluate_planetary(
        self,
        max_displacement=500,
        improvement_threshold_pct=5.0,
        min_translation_threshold=0.1,
        minimum_points=20,
    ):
        """Run pc_align against MOLA/LOLA and evaluate whether to keep it.

        Mirrors :meth:`align_and_evaluate` (the ICESat-2 path) but for
        planetary altimetry. Requires :meth:`load_planetary_csv` to have
        been called so ``self.planetary_points`` is populated.

        Decision logic:

        1. If fewer than ``minimum_points`` valid planetary points are
           available, return ``status="insufficient_points"``.
        2. Otherwise compute
           ``improvement_pct = (p50_beg - p50_end) / p50_beg * 100``. If
           ``p50_end >= p50_beg`` or
           ``improvement_pct <= improvement_threshold_pct`` or the
           translation magnitude is below ``min_translation_threshold ×
           DEM GSD``, delete the aligned DEM and return
           ``status="no_improvement"``.
        3. Otherwise re-run :meth:`planetary_to_dem_dh` to populate
           ``altimetry_minus_aligned_dem`` and return ``status="success"``.

        Parameters
        ----------
        max_displacement : float, optional
            ``--max-displacement`` for pc_align, in meters. Default 500
            (ASAP-Stereo's CTX cookbook recommendation).
        improvement_threshold_pct : float, optional
            Minimum p50 reduction (%) required to keep the aligned DEM.
        min_translation_threshold : float, optional
            Minimum translation magnitude as a fraction of the DEM GSD.
        minimum_points : int, optional
            Minimum number of valid altimetry points to attempt
            alignment. Planetary tracks are sparser than ICESat-2, so
            this defaults to a much smaller number than the Earth path.

        Returns
        -------
        AlignmentResult
        """
        from asp_plot.utils import detect_planetary_body

        body = detect_planetary_body(self.dem_fn)
        instrument = BODIES[body].altimetry_instrument

        parameters_used = {
            "max_displacement": max_displacement,
            "minimum_points": minimum_points,
            "min_translation_threshold": min_translation_threshold,
            "improvement_threshold_pct": improvement_threshold_pct,
        }

        if self.planetary_points is None or self.planetary_points.empty:
            return AlignmentResult(
                status="insufficient_points",
                alignment_report_df=pd.DataFrame(),
                aligned_dem_fn=None,
                improvement_pct=None,
                message=(
                    f"Alignment skipped: no {instrument} points loaded. "
                    "Call load_planetary_csv() first."
                ),
                parameters_used=parameters_used,
            )

        # planetary_to_dem_dh both samples and 3σ-filters; require enough
        # points whose dh is finite (i.e. fall inside the DEM extent).
        if "altimetry_minus_dem" not in self.planetary_points.columns:
            self.planetary_to_dem_dh()
        n_valid = int(self.planetary_points["altimetry_minus_dem"].notna().sum())
        if n_valid < minimum_points:
            self._remove_aligned_dem_if_present()
            return AlignmentResult(
                status="insufficient_points",
                alignment_report_df=pd.DataFrame(),
                aligned_dem_fn=None,
                improvement_pct=None,
                message=(
                    f"Alignment skipped: only {n_valid} {instrument} points "
                    f"overlap the DEM (need >= {minimum_points})."
                ),
                parameters_used=parameters_used,
            )

        csv_fn = self.to_csv_for_pc_align_planetary(
            filename_prefix=f"{instrument.lower()}_for_pc_align"
        )

        alignment = Alignment(self.directory, self.dem_fn)
        output_prefix = f"pc_align/pc_align_{instrument.lower()}"
        alignment.pc_align_dem_to_planetary_csv(
            planetary_csv=csv_fn,
            body=body,
            max_displacement=max_displacement,
            output_prefix=output_prefix,
        )

        report = alignment.pc_align_report(output_prefix=output_prefix)
        if not report:
            self._remove_aligned_dem_if_present()
            return AlignmentResult(
                status="insufficient_points",
                alignment_report_df=pd.DataFrame(),
                aligned_dem_fn=None,
                improvement_pct=None,
                message=(
                    "pc_align ran but produced no parseable log. Check "
                    f"{output_prefix}-log-pc_align*.txt."
                ),
                parameters_used=parameters_used,
            )

        df = pd.DataFrame([{"key": instrument.lower(), **report}])

        gsd = Raster(self.dem_fn).get_gsd()
        translation_too_small = (
            df["translation_magnitude"].iloc[0] < min_translation_threshold * gsd
        )

        p50_beg = float(report.get("p50_beg", float("nan")))
        p50_end = float(report.get("p50_end", float("nan")))
        improvement_pct = self._improvement_pct(p50_beg, p50_end)

        improvement_repr = (
            f"{improvement_pct:.1f}%" if improvement_pct is not None else "n/a"
        )
        if translation_too_small:
            reason = (
                f"Translation magnitude is below {min_translation_threshold*100:.0f}% "
                f"of the DEM GSD ({gsd:.2f} m), so no aligned DEM was "
                f"written despite a {improvement_repr} p50 reduction."
            )
        else:
            reason = (
                f"p50 {p50_beg:.2f} m -> {p50_end:.2f} m, "
                f"{improvement_repr} <= {improvement_threshold_pct:.1f}% "
                "threshold."
            )

        result = self._evaluate_improvement(
            df=df,
            p50_beg=p50_beg,
            p50_end=p50_end,
            improvement_pct=improvement_pct,
            improvement_threshold_pct=improvement_threshold_pct,
            translation_too_small=translation_too_small,
            no_improvement_reason=reason,
            parameters_used=parameters_used,
        )
        if result is not None:
            return result

        # Apply the translation and persist the aligned DEM
        aligned = alignment.apply_dem_translation(output_prefix=output_prefix)
        if aligned is None:
            self._remove_aligned_dem_if_present()
            return AlignmentResult(
                status="no_improvement",
                alignment_report_df=df,
                aligned_dem_fn=None,
                improvement_pct=improvement_pct,
                message=(
                    "pc_align reported an improvement but the translation "
                    "could not be applied. Aligned DEM not written."
                ),
                parameters_used=parameters_used,
            )

        self.aligned_dem_fn = aligned
        # Re-run planetary_to_dem_dh with n_sigma=None so we don't filter
        # the already-clean sample again.
        self.planetary_to_dem_dh(n_sigma=None)

        return self._success_result(
            df, p50_beg, p50_end, improvement_pct, parameters_used
        )

    def _remove_aligned_dem_if_present(self):
        """Delete the aligned DEM file if pc_align created one.

        Safe to call when no aligned DEM exists. Clears
        ``self.aligned_dem_fn`` on success.
        """
        aligned = getattr(self, "aligned_dem_fn", None)
        if aligned and os.path.exists(aligned):
            try:
                os.remove(aligned)
            except OSError as e:
                logger.warning(f"\nCould not remove aligned DEM {aligned}: {e}\n")
        self.aligned_dem_fn = None

    @staticmethod
    def _improvement_pct(p50_beg, p50_end):
        """Percent p50 reduction from alignment, or None if not computable.

        ``(p50_beg - p50_end) / p50_beg * 100``. Returns None when either
        value is non-finite or ``p50_beg`` is zero. Shared by the ICESat-2
        and planetary align-and-evaluate paths.
        """
        if not np.isfinite(p50_beg) or not np.isfinite(p50_end) or p50_beg == 0:
            return None
        return (p50_beg - p50_end) / p50_beg * 100.0

    def _evaluate_improvement(
        self,
        *,
        df,
        p50_beg,
        p50_end,
        improvement_pct,
        improvement_threshold_pct,
        translation_too_small,
        no_improvement_reason,
        parameters_used,
    ):
        """Shared keep/discard decision for ICESat-2 and planetary alignment.

        Applies the common rejection predicate used by both
        :meth:`align_and_evaluate` and :meth:`align_and_evaluate_planetary`:
        the aligned DEM is discarded when the p50 improvement is missing,
        non-positive, at or below ``improvement_threshold_pct``, or the
        translation was too small to write a DEM.

        Returns
        -------
        AlignmentResult or None
            A terminal ``no_improvement`` result (with the aligned DEM
            already cleaned up) when the alignment is rejected, or ``None``
            when it should be kept. In the keep case the caller performs its
            body-specific success finalization and builds the success result
            via :meth:`_success_result`.

        Notes
        -----
        ``no_improvement_reason`` is supplied by the caller because the
        Earth and planetary paths word the rejection differently (GSD value,
        translation wording). It is only consumed on the reject path.
        """
        if (
            improvement_pct is None
            or p50_end >= p50_beg
            or improvement_pct <= improvement_threshold_pct
            or translation_too_small
        ):
            self._remove_aligned_dem_if_present()
            return AlignmentResult(
                status="no_improvement",
                alignment_report_df=df,
                aligned_dem_fn=None,
                improvement_pct=improvement_pct,
                message=f"No significant improvement: {no_improvement_reason}",
                parameters_used=parameters_used,
            )
        return None

    def _success_result(self, df, p50_beg, p50_end, improvement_pct, parameters_used):
        """Build the shared ``success`` AlignmentResult.

        Identical for both bodies once the aligned DEM is retained on
        ``self.aligned_dem_fn``.
        """
        return AlignmentResult(
            status="success",
            alignment_report_df=df,
            aligned_dem_fn=self.aligned_dem_fn,
            improvement_pct=improvement_pct,
            message=(
                f"p50 improved from {p50_beg:.2f} m -> {p50_end:.2f} m "
                f"({improvement_pct:.1f}% reduction). Aligned DEM written to "
                f"{self.aligned_dem_fn}."
            ),
            parameters_used=parameters_used,
        )
