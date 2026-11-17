from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").rglob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_validates(path):
    # Cell ids are only allowed from nbformat 4.5, where they are required (#220, #221)
    nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    nbformat.validate(nb)
