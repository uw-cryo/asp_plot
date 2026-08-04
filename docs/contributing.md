# Contributing

## Install from source

Use either conda or [pixi](https://pixi.sh) — both give you the package in editable mode with the development and docs dependencies.

### conda

```bash
git clone git@github.com:uw-cryo/asp_plot.git
cd asp_plot
conda env create -f environment.yml
conda activate asp_plot
pre-commit install
```

The `environment.yml` installs the package in editable mode with development dependencies (`pip install -e ".[dev]"`).

If you want to rebuild the package, for instance while testing changes to the CLI tool, reinstall via:

```bash
pip install -e ".[dev]"
```

### pixi

Install pixi once per machine (`curl -fsSL https://pixi.sh/install.sh | sh`, `brew install pixi`, or `conda install -c conda-forge pixi`), then open a new shell so `~/.pixi/bin` is on your `PATH`. See the [installation guide](installation.md) for details and Windows instructions.

```bash
git clone git@github.com:uw-cryo/asp_plot.git
cd asp_plot
pixi run setup
```

Don't create a conda environment or virtualenv first — pixi replaces that step rather than running inside one, building a project-local environment in `.pixi/` that `pixi run` uses regardless of what is active in your shell. There is nothing to create or activate: every `pixi run` installs the environment first if it is missing, from the committed `pixi.lock`. The lockfile pins every resolved package per platform, so your environment matches everyone else's and matches CI.

The package itself is installed editable, so there is **no `pip install -e .` step, ever** — not even after adding or renaming a console script in `[project.scripts]`. pixi notices the manifest changed and re-syncs on the next `pixi run`, where the conda workflow needs an explicit reinstall.

The tasks defined in `pixi.toml` (list them with `pixi task list`):

| Task | What it does |
| --- | --- |
| `pixi run setup` | Install the pre-commit hooks |
| `pixi run test` | Run the test suite |
| `pixi run lint` | Run black, flake8, and isort over the whole repo |
| `pixi run docs` | Serve the docs locally with live reload |
| `pixi run docs-build` | Build the docs once into `docs/_build/html` |
| `pixi run lab` | Launch JupyterLab in the environment |
| `pixi run kernel` | Register the environment as a Jupyter kernel |

#### Adding a dependency

Runtime dependencies are declared in three places that must stay in sync: `pyproject.toml` (the source of truth for the released package), `pixi.toml`, and `conda-forge-recipe/meta.yaml`. Add the package to `pixi.toml` as well as `pyproject.toml`, run `pixi install`, and commit the updated `pixi.lock` alongside them.

```{warning}
Adding a dependency to `pyproject.toml` *only* appears to work — and quietly does the wrong thing. pixi will resolve it as a **PyPI wheel** rather than the conda-forge build, which is exactly the outcome `pixi.toml` exists to prevent: fine for a pure-Python package, but for anything compiled against GDAL or PROJ it is how you end up with a mismatched or unbuildable stack. Nothing warns you. Declaring it in `pixi.toml` is what keeps it coming from conda-forge.

CI will not catch this either. The `locked: true` check only fails when `pixi.lock` is out of date with the manifest, and the lock will already have been regenerated (see below), so a PyPI-resolved dependency sails straight through.
```

```{note}
**`pixi.lock` can change without you asking.** When the manifest no longer matches the lockfile, an ordinary `pixi run` — including `pixi run test` — regenerates it in place. So `git status` can come back dirty after a command you thought was read-only. That is pixi keeping the environment honest, not a bug; just check whether a lockfile change in your diff is one you intended before committing it.
```

**Please don't miss the pre-commit hooks** (`pre-commit install` or `pixi run setup`), which run linting prior to any commits using the `.pre-commit-config.yaml` file included in the repo.

## Run tests

```bash
pytest          # or: pixi run test
```

When you add a new feature, add some test coverage as well. Use `pytest -s` to see output during debugging.

## Run the example notebooks

The notebooks in `notebooks/` need a Jupyter kernel that can import `asp_plot`. A kernel is nothing more than a Python interpreter with `ipykernel` installed, so "choosing the right kernel" means pointing Jupyter at the interpreter that has the package — the conda environment, or with pixi the one in `.pixi/envs/default/`.

### With pixi

Either run JupyterLab from inside the environment, in which case the kernel is already correct and there is nothing to choose:

```bash
pixi run lab
```

Or, if you prefer your own JupyterLab or VS Code, register the environment once as a named kernel:

```bash
pixi run kernel
```

It then appears in the kernel picker as **asp_plot (pixi)**, alongside any conda environments, and points at `.pixi/envs/default/bin/python`. This writes a kernelspec to your user Jupyter directory (outside the repo); remove it with `jupyter kernelspec remove asp_plot-pixi`.

### With conda

Activate the environment and install `ipykernel` into it (`environment.yml` does not include it), then select the `asp_plot` environment as the kernel. Editors such as VS Code will usually offer to install `ipykernel` for you when you pick an environment that lacks it.

```{note}
If an early import fails in a notebook, it is almost always the kernel rather than the code: the notebook is running against an interpreter where `asp_plot` isn't installed. Check `import sys; print(sys.executable)` in the first cell — it should point inside `.pixi/envs/default/` or your conda environment.
```

## Add a feature

New to the codebase? `ARCHITECTURE.md` at the repository root is a module-by-module map of the package and its design patterns (registries, composition, the source/plotter splits) — the fastest way to find where your change belongs. `AGENTS.md` (also at the root) is the lean companion auto-discovered by AI coding agents (Claude Code and others): commands, gotchas, and process. Keep both in sync when you add or restructure a module.

Checkout main and pull to get the latest changes:

```bash
git checkout main
git pull
```

Create a feature branch:

```bash
git checkout -b my_feature
```

Make as many commits as you like while you work. When you are ready, submit the changes as a pull request.

After review, you may be asked to add tests for the new functionality. Add those in the `tests/` folder, and check that they work with:

```bash
pytest -s
```

When review is complete, [squash and merge](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/about-merge-methods-on-github#squashing-your-merge-commits) the changes to `main`, combining your commits into a single, descriptive commit.

## Versioning and CHANGELOG

This project follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Added functionality in a backwards compatible manner
- **PATCH**: Backwards compatible bug fixes and minor enhancements

All notable changes are documented in the [CHANGELOG](changelog.md). When contributing changes, please add an entry to the CHANGELOG.

## Release

To release a new version:

1. Update version in `pyproject.toml` following semantic versioning rules
2. Update `CHANGELOG.md` with the new version and date
3. Merge to `main`

The GitHub Actions workflow (`.github/workflows/release.yml`) automatically creates a GitHub Release, tag, and publishes to PyPI. The conda-forge feedstock picks up new PyPI versions automatically.
