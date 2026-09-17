#!/usr/bin/env python3
"""Downsample the PNG outputs embedded in the committed example notebooks.

Why this exists
---------------
The example notebooks are committed *with* their outputs so ReadTheDocs can
render them without the example datasets (see ``.readthedocs.yaml``). Those
outputs are base64 PNGs written by matplotlib at plot dpi, and they are
essentially the entire file: the largest example notebook is 28 MB, of which
21 KB is source and 27.9 MB is ``image/png``. Figures arrive up to 2931 px
wide, several times what any notebook viewer or docs page displays.

Re-running the notebooks at a lower dpi would need the full example datasets
and hours of ASP output, so this rewrites the already-committed outputs in
place instead. It touches nothing but the ``image/png`` payloads -- cell
source, execution counts, metadata, and every other MIME type are preserved
byte-for-byte, so the diff is limited to the base64 blobs.

Usage
-----
    python tools/shrink_notebook_outputs.py --check notebooks/
    python tools/shrink_notebook_outputs.py notebooks/

``--colors 256`` additionally quantizes to a 256-entry palette. That suits
matplotlib output well -- its colormaps are 256-entry LUTs to begin with -- but
it is lossy for multi-panel figures that combine several colormaps, so it is
off by default. Check the result before committing it.
"""

import argparse
import base64
import io
import json
import os
import sys

from PIL import Image

# Wide enough to stay sharp on a HiDPI display at the ~800 px that Jupyter and
# the nbsphinx docs theme actually lay an output image out at.
DEFAULT_MAX_WIDTH = 1600


def _iter_png_outputs(notebook):
    """Yield every output dict in a parsed notebook that carries an image/png."""
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                yield output


def _decode(payload):
    """Decode a notebook image/png payload, which may be a str or list of str."""
    if isinstance(payload, list):
        payload = "".join(payload)
    return base64.b64decode(payload)


def shrink_png(raw, max_width, colors):
    """Return a smaller encoding of ``raw``, or None if nothing was gained.

    Parameters
    ----------
    raw : bytes
        The decoded PNG.
    max_width : int
        Width ceiling in pixels. 0 disables resizing.
    colors : int
        Palette size for quantization. 0 disables quantization.

    Returns
    -------
    bytes or None
        The re-encoded PNG, or None when it came out no smaller than the
        original (PIL's encoder is not always a match for matplotlib's, so a
        no-op resize can genuinely inflate the file).
    """
    with Image.open(io.BytesIO(raw)) as img:
        img.load()

        if max_width and img.width > max_width:
            height = max(1, round(img.height * max_width / img.width))
            # LANCZOS because nearest/bilinear visibly chew up the 1 px axis
            # lines and antialiased tick labels these figures are full of.
            img = img.resize((max_width, height), Image.LANCZOS)

        if colors:
            img = img.convert("RGB").quantize(colors)

        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)

    return buf.getvalue() if buf.tell() < len(raw) else None


def process_notebook(path, max_width, colors, dry_run):
    """Shrink one notebook's outputs. Returns (bytes_before, bytes_after)."""
    before = os.path.getsize(path)
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)

    changed = False
    for output in _iter_png_outputs(notebook):
        raw = _decode(output["data"]["image/png"])
        smaller = shrink_png(raw, max_width, colors)
        if smaller is None:
            continue
        # nbformat writes image/png as a single base64 string with a trailing
        # newline; match that so unrelated notebooks do not churn.
        output["data"]["image/png"] = base64.b64encode(smaller).decode("ascii") + "\n"
        changed = True

    if not changed:
        return before, before

    # json.dump with indent=1 and a trailing newline is what nbformat writes,
    # so an untouched notebook round-trips to an empty diff.
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    if not dry_run:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return before, len(text.encode("utf-8"))


def find_notebooks(paths):
    """Expand files and directories into a sorted list of .ipynb paths."""
    found = []
    for path in paths:
        if os.path.isdir(path):
            for root, _, names in os.walk(path):
                if ".ipynb_checkpoints" in root:
                    continue
                found += [os.path.join(root, n) for n in names if n.endswith(".ipynb")]
        elif path.endswith(".ipynb"):
            found.append(path)
    return sorted(found)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        nargs="*",
        default=["notebooks"],
        help="Notebook files or directories to walk. Default: notebooks/",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=DEFAULT_MAX_WIDTH,
        help=f"Pixel width ceiling; 0 disables resizing. Default: {DEFAULT_MAX_WIDTH}",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=0,
        help="Quantize to this many palette colors (e.g. 256). 0 disables. Default: 0",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    args = parser.parse_args(argv)

    notebooks = find_notebooks(args.paths)
    if not notebooks:
        print("No notebooks found.", file=sys.stderr)
        return 1

    total_before = total_after = 0
    for path in notebooks:
        before, after = process_notebook(path, args.max_width, args.colors, args.check)
        total_before += before
        total_after += after
        if after != before:
            print(f"{before / 1048576:7.1f} -> {after / 1048576:6.1f} MB  {path}")

    saved = total_before - total_after
    print(
        f"\n{len(notebooks)} notebooks: {total_before / 1048576:.1f} -> "
        f"{total_after / 1048576:.1f} MB "
        f"({saved / 1048576:.1f} MB saved, {total_before / max(total_after, 1):.1f}x)"
    )
    if args.check:
        print("--check: nothing written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
