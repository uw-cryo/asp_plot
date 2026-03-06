"""Sphinx configuration for asp_plot documentation."""

import importlib.metadata

project = "asp_plot"
copyright = "2026, UW Cryosphere"
author = "Ben Purinton, David Shean, Shashank Bhushan"
try:
    version = release = importlib.metadata.version("asp_plot")
except importlib.metadata.PackageNotFoundError:
    version = release = "dev"

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
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/uw-cryo/asp_plot",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "show_toc_level": 2,
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
html_extra_path = ["_extra"]
templates_path = ["_templates"]
