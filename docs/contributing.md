# Contributing

## Install from source

```bash
git clone git@github.com:uw-cryo/asp_plot.git
cd asp_plot
conda env create -f environment.yml
conda activate asp_plot
pre-commit install
```

The `environment.yml` installs the package in editable mode with development dependencies (`pip install -e ".[dev]"`).

**Please don't miss the `pre-commit install` step**, which runs linting prior to any commits using the `.pre-commit-config.yaml` file included in the repo.

If you want to rebuild the package, for instance while testing changes to the CLI tool, reinstall via:

```bash
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

When you add a new feature, add some test coverage as well. Use `pytest -s` to see output during debugging.

## Add a feature

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
