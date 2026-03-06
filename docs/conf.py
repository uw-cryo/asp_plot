"""Sphinx configuration for asp_plot documentation."""

import importlib.metadata

project = "asp_plot"
copyright = "2026, UW Cryosphere"
author = "Ben Purinton, David Shean, Shashank Bhushan"
version = release = importlib.metadata.version("asp_plot")

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
]

# -- MyST / Notebook settings -----------------------------------------------
myst_enable_extensions = ["colon_fence"]
nb_execution_mode = "off"

# -- AutoAPI settings (static analysis, no import needed) --------------------
autoapi_dirs = ["../asp_plot"]
autoapi_type = "python"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_member_order = "groupwise"
suppress_warnings = ["autoapi"]

# -- Theme -------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/uw-cryo/asp_plot",
    "use_edit_page_button": True,
    "show_toc_level": 2,
}
html_context = {
    "github_user": "uw-cryo",
    "github_repo": "asp_plot",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- General -----------------------------------------------------------------
exclude_patterns = ["_build", "**.ipynb_checkpoints", ".DS_Store"]
html_static_path = ["_static"]
