"""
Versioned adapter for parsing NASA Ames Stereo Pipeline (ASP) log files.

ASP writes one plain-text log per tool invocation (``bundle_adjust``,
``stereo_pprc`` ... ``stereo_tri``, ``point2dem``). Each log starts with a
version banner, then the literal command line that was run, then timestamped
``console`` lines. The exact layout has been stable across ASP 3.x but is not
guaranteed to stay fixed.

Rather than scatter hardcoded regexes and ``line.split()[0]`` indexing through
the call sites (which fail silently when the format drifts), all format
knowledge lives here behind a small adapter that is *keyed by ASP version*. A
new ASP log layout becomes a new :class:`AspLogFormat` subclass registered in
``ASP_LOG_FORMATS``; everything downstream keeps working through the stable
:class:`AspLog` interface.

When a field cannot be parsed the adapter returns ``None`` and logs a warning
(at ``logger`` level) so format drift surfaces instead of being swallowed.
"""

import logging
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)


# The ordered stages of an ASP stereo run. Used to pick the earliest- and
# latest-running stereo logs for run-time spans without hardcoding "pprc" and
# "tri" at the call site -- if either is absent we fall back to the nearest
# present stage.
STEREO_STEP_ORDER = (
    "stereo_pprc",
    "stereo_corr",
    "stereo_blend",
    "stereo_rfne",
    "stereo_fltr",
    "stereo_tri",
)

# Known ASP command-line tools whose invocation may appear as the first token
# of a log's command line. Used to locate the command line robustly (by the
# executable basename) instead of by an arbitrary substring match.
ASP_TOOL_NAMES = frozenset(
    {
        "bundle_adjust",
        "parallel_bundle_adjust",
        "jitter_solve",
        "stereo",
        "parallel_stereo",
        "point2dem",
        "pc_align",
        "dem_mosaic",
        "mapproject",
        *STEREO_STEP_ORDER,
    }
)


class AspLogFormat:
    """Adapter describing how to read one ASP log layout.

    The base class implements the ASP 3.x layout. To support a drifted format,
    subclass it, override the class attributes (or methods) that changed, give
    it a distinguishing :meth:`supports`, and register the subclass in
    ``ASP_LOG_FORMATS`` ahead of the default.
    """

    #: Short identifier, surfaced in logs.
    name = "asp-3.x"

    #: Banner on the first line, e.g. ``ASP 3.4.0-alpha``.
    version_banner_re = re.compile(r"^ASP\s+(?P<version>\S+)")

    #: A leading ``YYYY-MM-DD HH:MM:SS`` timestamp on a console line.
    timestamp_re = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
    timestamp_fmt = "%Y-%m-%d %H:%M:%S"

    #: Reference-DEM announcements. ASP phrases this differently per tool
    #: (``Loading DEM:`` in bundle_adjust, ``Using input DEM:`` in stereo), so
    #: one case-insensitive pattern covers the known variants.
    reference_dem_re = re.compile(
        r"(?:Loading DEM|Using input DEM|Input DEM)\s*:\s*(?P<dem>\S+)",
        re.IGNORECASE,
    )

    @classmethod
    def supports(cls, banner_line):
        """Whether this adapter handles a log with the given first line.

        The default ASP 3.x adapter claims any ``ASP 3.*`` banner.
        """
        match = cls.version_banner_re.match(banner_line or "")
        if not match:
            return False
        return match.group("version").startswith("3.")

    def parse_version(self, banner_line):
        """Return the ASP version string from the banner, or ``None``."""
        match = self.version_banner_re.match(banner_line or "")
        if not match:
            return None
        return match.group("version")

    def parse_timestamp(self, line):
        """Return the :class:`datetime` from a leading timestamp, or ``None``."""
        match = self.timestamp_re.match(line or "")
        if not match:
            return None
        try:
            return datetime.strptime(match.group("ts"), self.timestamp_fmt)
        except ValueError:
            logger.warning(
                "%s: could not parse timestamp %r with format %r",
                self.name,
                match.group("ts"),
                self.timestamp_fmt,
            )
            return None

    def parse_command_line(self, lines):
        """Locate the tool invocation line in ``lines``.

        Returns ``(tool, command)`` where ``tool`` is the executable basename
        (e.g. ``stereo_tri``) and ``command`` is the full stripped command
        line, or ``None`` if no known ASP tool invocation is found.

        The line is found by matching the basename of its first token against
        :data:`ASP_TOOL_NAMES`, skipping the banner, build metadata and
        timestamped console lines -- robust against substring collisions in
        later log output.
        """
        for raw in lines:
            stripped = raw.strip()
            if not stripped or self.timestamp_re.match(stripped):
                continue
            first_token = stripped.split()[0]
            tool = os.path.basename(first_token)
            if tool in ASP_TOOL_NAMES:
                return tool, stripped
        return None

    def parse_reference_dem(self, lines):
        """Return the last reference-DEM path announced in ``lines``, or ``None``.

        The last match wins to mirror the historical behavior (ASP re-announces
        the DEM and the final mention is the one used).
        """
        reference_dem = None
        for line in lines:
            match = self.reference_dem_re.search(line)
            if match:
                reference_dem = match.group("dem").strip()
        return reference_dem


def register_format(fmt, prepend=True):
    """Register an :class:`AspLogFormat` subclass for adapter selection.

    The extension point for ASP format drift: when a future ASP version
    changes the log layout, subclass :class:`AspLogFormat`, override what
    changed and its :meth:`~AspLogFormat.supports`, then register it here.
    Registered formats are consulted by :func:`select_format` in order, so by
    default a new (more specific) format is prepended ahead of the permissive
    ASP 3.x default.
    """
    if prepend:
        ASP_LOG_FORMATS.insert(0, fmt)
    else:
        ASP_LOG_FORMATS.append(fmt)


def select_format(banner_line):
    """Return an adapter instance for a log whose first line is ``banner_line``.

    Falls back to the default ASP 3.x adapter (with a warning) when no
    registered format claims the banner, so an unrecognized version degrades
    gracefully instead of dropping all fields.
    """
    for fmt in ASP_LOG_FORMATS:
        if fmt.supports(banner_line):
            return fmt()
    logger.warning(
        "Unrecognized ASP log banner %r; falling back to %s parsing.",
        (banner_line or "").strip(),
        DEFAULT_FORMAT.name,
    )
    return DEFAULT_FORMAT()


class AspLog:
    """A single ASP log file, parsed through the matching versioned adapter.

    Reads the file once on construction and exposes the parsed fields as
    properties. Selecting the adapter from the version banner means a caller
    never needs to know which ASP version produced the log.
    """

    def __init__(self, path):
        self.path = path
        with open(path, "r") as f:
            self.lines = f.read().splitlines()
        banner = self.lines[0] if self.lines else ""
        self.format = select_format(banner)

    @property
    def asp_version(self):
        """The ASP version string, or ``None`` if the banner is absent."""
        if not self.lines:
            return None
        return self.format.parse_version(self.lines[0])

    @property
    def command(self):
        """The full command line (tool + args), or ``None`` if not found."""
        result = self.format.parse_command_line(self.lines)
        if result is None:
            logger.warning("No ASP tool command line found in %s", self.path)
            return None
        return result[1]

    @property
    def tool(self):
        """The executable basename of the invocation (e.g. ``stereo_tri``)."""
        result = self.format.parse_command_line(self.lines)
        return result[0] if result else None

    def canonical_command(self, tool_name):
        """Return the command line with its executable replaced by ``tool_name``.

        ASP logs the executable as an absolute path (or a stage-specific name
        like ``stereo_tri``); reports want a canonical, path-free invocation
        such as ``bundle_adjust ...`` or ``stereo ...``. Returns ``None`` if no
        command line was found.
        """
        command = self.command
        if command is None:
            return None
        parts = command.split(maxsplit=1)
        args = parts[1] if len(parts) > 1 else ""
        return f"{tool_name} {args}".rstrip()

    @property
    def timestamps(self):
        """All parsed console timestamps, in file order."""
        out = []
        for line in self.lines:
            ts = self.format.parse_timestamp(line)
            if ts is not None:
                out.append(ts)
        return out

    @property
    def first_timestamp(self):
        """The earliest console timestamp, or ``None``."""
        timestamps = self.timestamps
        return timestamps[0] if timestamps else None

    @property
    def last_timestamp(self):
        """The latest console timestamp, or ``None``."""
        timestamps = self.timestamps
        return timestamps[-1] if timestamps else None

    @property
    def reference_dem(self):
        """The reference-DEM path announced in the log, or ``None``."""
        return self.format.parse_reference_dem(self.lines)


# Registered formats, tried in order; the first whose ``supports`` returns True
# wins. Use :func:`register_format` to add more specific formats ahead of the
# permissive ASP 3.x default.
ASP_LOG_FORMATS = [AspLogFormat]
DEFAULT_FORMAT = AspLogFormat
