from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("asp_plot")
except PackageNotFoundError:
    __version__ = "unknown"
