#!/usr/bin/env bash
# Fetch the example reports into docs/_static/reports/ for the docs build.
#
# The reports are published as GitHub Release assets rather than committed to
# git: PDFs do not delta-compress, so every regeneration became a permanent new
# blob and seven files had grown into 846 MB of repository history (issue #201).
#
# GitHub serves release assets as content-disposition: attachment, so the
# iframes on docs/examples/reports.md cannot point at them directly — the build
# downloads them and self-hosts them under _static/, which is what the
# `cp reports/*.pdf` line used to do.
#
# When the reports are regenerated: publish a new release, then bump the one
# tag below. Publish the release *before* pushing the commit that points at it,
# or the docs build fails on a missing asset. See AGENTS.md.
set -euo pipefail

REPORTS_RELEASE="reports-2026-09-18"
BASE="https://github.com/uw-cryo/asp_plot/releases/download/${REPORTS_RELEASE}"
OUT="$(cd "$(dirname "$0")" && pwd)/_static/reports"

REPORTS=(
    WorldView_Atlanta_MVS
    WorldView_UCSD
    ASTER
    ASTER_mapproj
    LRO_NAC
    MOC
    MOC_mapproj
)

mkdir -p "$OUT"
echo "Fetching example reports from ${REPORTS_RELEASE}"
for name in "${REPORTS[@]}"; do
    # -f so a missing or renamed asset fails the build loudly, rather than
    # leaving a 404 body on disk and shipping a broken iframe.
    curl -fsSL --retry 3 -o "${OUT}/${name}-asp-report.pdf" "${BASE}/${name}-asp-report.pdf"
    echo "  ${name}-asp-report.pdf"
done
