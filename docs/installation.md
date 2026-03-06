# Installation

## conda (recommended)

Install `asp_plot` and all dependencies in one step:

```bash
conda install -c conda-forge asp-plot
```

## pip

Alternatively, install with pip:

```bash
pip install asp-plot
```

```{note}
Some dependencies (notably GDAL) can be difficult to install via pip alone. If you run into issues, use the conda approach above, or create a conda environment first:

    conda env create -f environment.yml
    conda activate asp_plot
    pip install asp-plot
```

## Install from source (development)

For contributing to the project or modifying the source code:

```bash
git clone git@github.com:uw-cryo/asp_plot.git
cd asp_plot
conda env create -f environment.yml
conda activate asp_plot
pre-commit install
```

The `environment.yml` installs the package in editable mode with development dependencies (`pip install -e ".[dev]"`).

See the [Contributing](contributing.md) guide for more details on the development workflow.
