import re
from datetime import datetime

import pytest

from asp_plot import asp_log
from asp_plot.asp_log import AspLog, AspLogFormat, select_format

BA_LOG = "tests/test_data/ba/log-bundle_adjust.txt"
PPRC_LOG = "tests/test_data/stereo/log-stereo_pprc.txt"
TRI_LOG = "tests/test_data/stereo/log-stereo_tri.txt"
POINT2DEM_LOG = "tests/test_data/stereo/log-point2dem.txt"

VARIANT_LOG = "tests/test_data/asp_log_variant/log-bundle_adjust-variant.txt"


class TestAspLog3x:
    """Adapter behavior against the real ASP 3.4.0-alpha fixtures."""

    def test_version(self):
        assert AspLog(BA_LOG).asp_version == "3.4.0-alpha"
        assert AspLog(PPRC_LOG).asp_version == "3.4.0-alpha"

    def test_tool_detection_by_basename(self):
        # bundle_adjust is logged as an absolute path; the executable basename
        # is what identifies it -- not an arbitrary substring.
        assert AspLog(BA_LOG).tool == "bundle_adjust"
        assert AspLog(TRI_LOG).tool == "stereo_tri"
        assert AspLog(POINT2DEM_LOG).tool == "point2dem"

    def test_canonical_command_strips_executable_path(self):
        cmd = AspLog(BA_LOG).canonical_command("bundle_adjust")
        assert cmd.startswith("bundle_adjust -t dg ")
        # No absolute executable path leaks into the canonical command.
        assert "/libexec/bundle_adjust" not in cmd

    def test_canonical_command_renames_stage(self):
        # stereo_tri's invocation is canonicalized to a path-free "stereo ...".
        cmd = AspLog(TRI_LOG).canonical_command("stereo")
        assert cmd.startswith("stereo --stereo-algorithm asp_mgm ")

    def test_timestamps_first_last(self):
        log = AspLog(PPRC_LOG)
        assert log.first_timestamp == datetime(2024, 4, 14, 17, 14, 31)
        assert log.last_timestamp == datetime(2024, 4, 14, 17, 55, 43)
        assert log.first_timestamp <= log.last_timestamp

    def test_reference_dem_loading(self):
        # bundle_adjust announces "Loading DEM:".
        assert (
            AspLog(BA_LOG).reference_dem
            == "/nobackup/bpurint1/data/utqiagvik/COP/COP30_utqiagvik_lzw-adj_proj.tif"
        )

    def test_reference_dem_using_input(self):
        # stereo_pprc announces "Using input DEM:" -- a different phrasing that
        # the single unified pattern still handles.
        assert (
            AspLog(PPRC_LOG).reference_dem
            == "/nobackup/bpurint1/data/utqiagvik/COP/COP30_utqiagvik_lzw-adj_proj.tif"
        )

    def test_format_selected_is_3x(self):
        assert isinstance(select_format("ASP 3.4.0-alpha"), AspLogFormat)
        assert select_format("ASP 3.4.0-alpha").name == "asp-3.x"


class TestFallback:
    def test_unrecognized_banner_falls_back_to_default(self):
        # An unknown banner does not raise; it degrades to the default adapter.
        fmt = select_format("totally unexpected first line")
        assert isinstance(fmt, AspLogFormat)

    def test_missing_command_returns_none_not_raise(self, tmp_path):
        log_path = tmp_path / "log-empty.txt"
        log_path.write_text("ASP 3.4.0-alpha\nBuild ID: x\n\nno command here\n")
        log = AspLog(str(log_path))
        assert log.command is None
        assert log.canonical_command("stereo") is None


# --- Format-variant adapter: proves the registry/dispatch is not vacuous. ---
#
# A synthetic future "ASP-NG" layout that drifts in three ways at once:
#   * banner word ("ASP-NG" instead of "ASP")
#   * ISO-8601 timestamps with a "T" separator
#   * reference DEM phrased as "Reference DEM:"
# Only the changed pieces are overridden, demonstrating the intended extension
# pattern for real ASP format drift.
class _VariantFormat(AspLogFormat):
    name = "asp-ng-4.x"
    version_banner_re = re.compile(r"^ASP-NG\s+(?P<version>\S+)")
    timestamp_re = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    timestamp_fmt = "%Y-%m-%dT%H:%M:%S"
    reference_dem_re = re.compile(r"Reference DEM\s*:\s*(?P<dem>\S+)", re.IGNORECASE)

    @classmethod
    def supports(cls, banner_line):
        match = cls.version_banner_re.match(banner_line or "")
        return bool(match) and match.group("version").startswith("4.")


@pytest.fixture
def variant_registered():
    asp_log.register_format(_VariantFormat)
    try:
        yield
    finally:
        asp_log.ASP_LOG_FORMATS.remove(_VariantFormat)


class TestFormatVariant:
    def test_default_does_not_claim_variant_banner(self):
        # Without the variant adapter registered, the ASP-NG banner is
        # unrecognized and falls back to the default (no version parsed).
        assert AspLogFormat.supports("ASP-NG 4.1.0") is False

    def test_variant_adapter_selected_and_parses(self, variant_registered):
        log = AspLog(VARIANT_LOG)
        assert log.format.name == "asp-ng-4.x"
        assert log.asp_version == "4.1.0"
        # ISO-8601 "T"-separated timestamps parse through the overridden format.
        assert log.first_timestamp == datetime(2025, 6, 1, 8, 0, 0)
        assert log.last_timestamp == datetime(2025, 6, 1, 8, 7, 30)
        # "Reference DEM:" phrasing handled by the overridden pattern.
        assert log.reference_dem == "/data/ref/cop30_variant.tif"
        # Tool detection by basename is format-independent.
        assert log.tool == "bundle_adjust"
        assert log.canonical_command("bundle_adjust").startswith("bundle_adjust -t dg ")

    def test_registration_is_isolated(self, variant_registered):
        # Inside the fixture the variant is registered...
        assert _VariantFormat in asp_log.ASP_LOG_FORMATS
