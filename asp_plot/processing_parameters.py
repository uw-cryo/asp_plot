import glob
import logging
import os

from asp_plot.asp_log import STEREO_STEP_ORDER, AspLog
from asp_plot.utils import glob_file

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ProcessingParameters:
    """
    Extract and manage processing parameters from ASP log files.

    This class extracts parameters from ASP processing log files,
    including bundle adjustment, stereo, and point2dem logs. It provides
    methods to parse these logs and obtain command lines, run times, and
    other relevant processing information.

    All of the format-specific parsing (version banner, command line,
    timestamps, reference DEM) is delegated to the versioned adapter in
    :mod:`asp_plot.asp_log`, so this class stays focused on locating the right
    log files and assembling the parameter dictionary.

    Attributes
    ----------
    processing_directory : str
        Root directory of ASP processing
    bundle_adjust_directory : str or None
        Subdirectory containing bundle adjustment outputs
    stereo_directory : str or None
        Subdirectory containing stereo outputs
    full_ba_directory : str or None
        Full path to bundle adjustment directory
    full_stereo_directory : str or None
        Full path to stereo directory
    bundle_adjust_log : str or None
        Path to bundle adjustment log file
    stereo_logs : list or None
        List of paths to stereo log files
    point2dem_log : str or None
        Path to point2dem log file
    processing_parameters_dict : dict
        Dictionary to store extracted parameters

    Examples
    --------
    >>> params = ProcessingParameters('/path/to/asp', 'ba', 'stereo')
    >>> params_dict = params.from_log_files()
    >>> print(params_dict['bundle_adjust'])  # Print bundle adjustment command line
    >>> print(params_dict['stereo_run_time'])  # Print stereo processing time
    """

    def __init__(
        self, processing_directory, bundle_adjust_directory=None, stereo_directory=None
    ):
        """
        Initialize the ProcessingParameters object.

        Parameters
        ----------
        processing_directory : str
            Root directory of ASP processing
        bundle_adjust_directory : str, optional
            Subdirectory containing bundle adjustment outputs, default is None
        stereo_directory : str, optional
            Subdirectory containing stereo outputs, default is None

        Notes
        -----
        This constructor attempts to locate the relevant log files in the
        specified directories. If directories or log files are not found,
        the corresponding attributes are set to None.
        """
        self.processing_directory = os.path.expanduser(processing_directory)
        self.bundle_adjust_directory = bundle_adjust_directory
        self.stereo_directory = stereo_directory
        self.full_ba_directory = (
            os.path.join(self.processing_directory, bundle_adjust_directory)
            if bundle_adjust_directory
            else None
        )
        self.full_stereo_directory = (
            os.path.join(self.processing_directory, stereo_directory)
            if stereo_directory
            else None
        )
        self.processing_parameters_dict = {}

        # Locate log files. A missing directory (None) means that stage was not
        # requested -- not an error -- so we guard on it rather than catching a
        # blanket exception that would also hide real I/O problems.
        self.bundle_adjust_log = (
            glob_file(self.full_ba_directory, "*log-bundle_adjust*.txt")
            if self.full_ba_directory
            else None
        )
        self.stereo_logs = (
            glob.glob(os.path.join(self.full_stereo_directory, "*log-stereo*.txt"))
            if self.full_stereo_directory
            else None
        )
        self.point2dem_log = (
            glob_file(self.full_stereo_directory, "*log-point2dem*.txt")
            if self.full_stereo_directory
            else None
        )

    def _stereo_log(self, step):
        """Return the stereo log for a given pipeline ``step``, or None.

        ``step`` is a name from :data:`asp_plot.asp_log.STEREO_STEP_ORDER`
        (e.g. ``"stereo_pprc"``).
        """
        if not self.stereo_logs:
            return None
        return next(
            (log for log in self.stereo_logs if f"log-{step}" in log),
            None,
        )

    def _ordered_stereo_logs(self):
        """Return present stereo logs as ``(step, path)`` in pipeline order."""
        ordered = []
        for step in STEREO_STEP_ORDER:
            log = self._stereo_log(step)
            if log:
                ordered.append((step, log))
        return ordered

    def get_asp_version(self):
        """Extract the ASP version string from the first available log file.

        Returns
        -------
        str
            ASP version string (e.g., "3.4.0-alpha"), or "N/A" if not found.
        """
        log_candidates = []
        if self.bundle_adjust_log:
            log_candidates.append(self.bundle_adjust_log)
        pprc = self._stereo_log("stereo_pprc")
        if pprc:
            log_candidates.append(pprc)
        if self.point2dem_log:
            log_candidates.append(self.point2dem_log)

        for log_file in log_candidates:
            try:
                version = AspLog(log_file).asp_version
            except OSError as e:
                logger.warning("Could not read %s for ASP version: %s", log_file, e)
                continue
            if version:
                return version
        return "N/A"

    def from_log_files(self):
        """
        Extract processing parameters from log files.

        Parses the bundle adjustment, stereo, and point2dem log files
        to extract command lines, run times, and other parameters.

        Returns
        -------
        dict
            Dictionary containing processing parameters:
            - processing_timestamp: When processing was performed
            - reference_dem: Path to reference DEM
            - bundle_adjust: Bundle adjustment command line
            - bundle_adjust_run_time: Time to run bundle adjustment
            - stereo: Stereo command line
            - stereo_run_time: Time to run stereo
            - point2dem: point2dem command line
            - point2dem_run_time: Time to run point2dem

        Notes
        -----
        If bundle adjustment was not run, its command and run time are set
        to placeholder values. Reference DEM is searched for in either
        bundle adjustment or stereo logs, depending on availability.
        """
        if self.bundle_adjust_directory:
            bundle_adjust_params, ba_run_time, reference_dem = (
                self.from_bundle_adjust_log()
            )
            if reference_dem:
                processing_timestamp, stereo_params, stereo_run_time = (
                    self.from_stereo_log()
                )
            else:
                processing_timestamp, stereo_params, stereo_run_time, reference_dem = (
                    self.from_stereo_log(search_for_reference_dem=True)
                )
        else:
            bundle_adjust_params = "Bundle adjustment not run"
            ba_run_time = "N/A"
            processing_timestamp, stereo_params, stereo_run_time, reference_dem = (
                self.from_stereo_log(search_for_reference_dem=True)
            )

        point2dem_params, point2dem_run_time = self.from_point2dem_log()

        self.processing_parameters_dict = {
            "asp_version": self.get_asp_version(),
            "processing_timestamp": processing_timestamp,
            "reference_dem": reference_dem,
            "bundle_adjust": bundle_adjust_params,
            "bundle_adjust_run_time": ba_run_time,
            "stereo": stereo_params,
            "stereo_run_time": stereo_run_time,
            "point2dem": point2dem_params,
            "point2dem_run_time": point2dem_run_time,
        }

        return self.processing_parameters_dict

    def from_bundle_adjust_log(self):
        """
        Extract parameters from the bundle adjustment log file.

        Parses the bundle adjustment log file to extract the command line,
        identifies the reference DEM, and calculates the run time.

        Returns
        -------
        tuple
            A tuple containing (bundle_adjust_params, run_time, reference_dem)
            where:
            - bundle_adjust_params: str, the canonical ``bundle_adjust`` command
            - run_time: str, formatted run time (e.g., "2 hours and 15 minutes")
            - reference_dem: str, path to the reference DEM used ("" if none)

        Raises
        ------
        ValueError
            If the bundle adjustment log file cannot be found
        """
        if not self.bundle_adjust_log:
            raise ValueError(
                f"\n\nCould not find bundle adjust log file in {self.full_ba_directory}\nCheck that the *log*.txt file exists in the directory specified.\n\n"
            )
        log = AspLog(self.bundle_adjust_log)
        bundle_adjust_params = log.canonical_command("bundle_adjust")
        reference_dem = log.reference_dem or ""
        run_time = self.get_run_time([self.bundle_adjust_log])

        return bundle_adjust_params, run_time, reference_dem

    def from_point2dem_log(self):
        """
        Extract parameters from the point2dem log file.

        Parses the point2dem log file to extract the command line
        and calculates the run time.

        Returns
        -------
        tuple
            A tuple containing (point2dem_params, run_time) where:
            - point2dem_params: str, the canonical ``point2dem`` command line
            - run_time: str, formatted run time (e.g., "0 hours and 45 minutes")

        Raises
        ------
        ValueError
            If the point2dem log file cannot be found
        """
        if not self.point2dem_log:
            raise ValueError(
                f"\n\nCould not find point2dem log file in {self.full_stereo_directory}\nCheck that the *log*.txt file exists in the directory specified.\n\n"
            )
        point2dem_params = AspLog(self.point2dem_log).canonical_command("point2dem")
        run_time = self.get_run_time([self.point2dem_log])

        return point2dem_params, run_time

    def from_stereo_log(self, search_for_reference_dem=False):
        """
        Extract parameters from the stereo log files.

        Parses the stereo log files to extract command lines, processing timestamp,
        and optionally searches for reference DEM information. The earliest and
        latest stereo stages present (per
        :data:`asp_plot.asp_log.STEREO_STEP_ORDER`) bracket the run, so the
        command/reference DEM come from the final stage and the run-time span
        from first-stage start to last-stage end.

        Parameters
        ----------
        search_for_reference_dem : bool, optional
            Whether to search for reference DEM in the log files, default is False

        Returns
        -------
        tuple
            If search_for_reference_dem is False, returns:
            (processing_timestamp, stereo_params, run_time) where:
            - processing_timestamp: datetime, when processing started
            - stereo_params: str, the canonical ``stereo`` command line
            - run_time: str, formatted run time

            If search_for_reference_dem is True, additionally returns:
            - reference_dem: str, path to the reference DEM used ("" if none)

        Raises
        ------
        ValueError
            If stereo log files cannot be found

        Notes
        -----
        The stereo process proceeds through several steps:
        1. stereo_pprc - preprocessing
        2. stereo_corr - correlation
        3. stereo_blend - tile blending (logs in tile/ dirs)
        4. stereo_rfne - refinement (logs in tile/ dirs)
        5. stereo_fltr - filtering
        6. stereo_tri - triangulation

        This method uses the first stage's log for the start time and the last
        stage's log for the end time to calculate total runtime.
        """
        if not self.stereo_logs:
            raise ValueError(
                f"\n\nCould not find stereo log files in {self.full_stereo_directory}\nCheck that these *log*.txt files exist in the directory specified.\n\n"
            )
        ordered = self._ordered_stereo_logs()
        if not ordered:
            raise ValueError(
                f"\n\nCould not match any known stereo stage in {self.full_stereo_directory}\nExpected logs such as log-stereo_pprc*.txt ... log-stereo_tri*.txt.\n\n"
            )
        first_log = ordered[0][1]
        last_log = ordered[-1][1]

        last_stage_log = AspLog(last_log)
        stereo_params = last_stage_log.canonical_command("stereo")

        # processing_timestamp mirrors historical behavior: the last console
        # timestamp recorded in the first stage's (pprc) log.
        processing_timestamp = AspLog(first_log).last_timestamp or ""

        run_time = self.get_run_time([first_log, last_log])

        if search_for_reference_dem:
            reference_dem = last_stage_log.reference_dem
            if not reference_dem:
                # Fall back to the first stage (pprc announces "Using input DEM:").
                reference_dem = AspLog(first_log).reference_dem or ""
            return processing_timestamp, stereo_params, run_time, reference_dem
        else:
            return processing_timestamp, stereo_params, run_time

    def get_reference_dem(self, logfile):
        """
        Extract reference DEM path from a log file.

        Parameters
        ----------
        logfile : str
            Path to the log file to search

        Returns
        -------
        str
            Path to the reference DEM if found, empty string otherwise

        Notes
        -----
        Parsing is delegated to the versioned adapter, which recognizes the
        known ASP phrasings ("Loading DEM:", "Using input DEM:", "Input DEM:").
        """
        return AspLog(logfile).reference_dem or ""

    def get_run_time(self, logfiles):
        """
        Calculate the run time from timestamps in log files.

        Extracts the earliest timestamp from the first log file and the
        latest timestamp from the last log file to compute the total
        execution time.

        Parameters
        ----------
        logfiles : list of str
            Paths to log files to analyze. The first file is used to find
            the start time, and the last file is used to find the end time.

        Returns
        -------
        str
            Formatted run time as "X hours and Y minutes", or "N/A" if
            timestamps couldn't be found

        Notes
        -----
        This is a helper method used by other methods to calculate
        processing run times. It expects log files to have timestamps
        in the format "YYYY-MM-DD HH:MM:SS".

        If only one log file is provided, it will be used for both
        start and end times. This is useful for single-stage processes.
        """
        start_log = logfiles[0]
        end_log = logfiles[-1] if len(logfiles) > 1 else start_log

        start_time = AspLog(start_log).first_timestamp
        end_time = AspLog(end_log).last_timestamp

        if start_time and end_time:
            time_diff = end_time - start_time
            hours, remainder = divmod(time_diff.total_seconds(), 3600)
            minutes = remainder // 60
            run_time = f"{int(hours)} hours and {int(minutes)} minutes"
        else:
            run_time = "N/A"

        return run_time
