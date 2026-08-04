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

For contributing to the project or modifying the source code, use either conda or [pixi](https://pixi.sh). Both produce an equivalent environment with the package installed in editable mode.

### conda

```bash
git clone git@github.com:uw-cryo/asp_plot.git
cd asp_plot
conda env create -f environment.yml
conda activate asp_plot
pre-commit install
```

The `environment.yml` installs the package in editable mode with development dependencies (`pip install -e ".[dev]"`).

### pixi

First install pixi itself — once per machine, not per project. Any of these work:

```bash
curl -fsSL https://pixi.sh/install.sh | sh    # macOS / Linux
brew install pixi                             # macOS, via Homebrew
conda install -c conda-forge pixi             # if you already have conda
```

```{note}
The install script places the binary in `~/.pixi/bin` and appends that directory to your `PATH` via your shell profile. **Open a new shell** (or `source ~/.zshrc`) before `pixi` will resolve — otherwise you'll get `command not found: pixi` even though the install succeeded. Windows instructions are in the [pixi installation docs](https://pixi.sh/latest/#installation).
```

Then:

```bash
git clone git@github.com:uw-cryo/asp_plot.git
cd asp_plot
pixi run setup
```

`pixi` needs no separate environment creation or activation step: any `pixi run` command installs the environment first if it is missing. Here `pixi run setup` installs the pre-commit hooks, and installs the environment on the way. Use `pixi shell` to work inside it interactively, or prefix individual commands, for example `pixi run asp_report --help`.

The difference from conda is `pixi.lock`, a committed lockfile pinning every resolved package for each supported platform (`linux-64`, `osx-arm64`, `osx-64`), so environments don't drift between machines or between a contributor and CI.

See the [Contributing](contributing.md) guide for more details on the development workflow.
