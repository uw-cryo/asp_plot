"""Abstract base class and shared helpers for sensor metadata readers.

See the :mod:`asp_plot.sensors` package docstring for the sensor-agnostic
scene-dict schema that concrete readers produce.
"""

import os
from abc import ABC, abstractmethod


class SensorMetadata(ABC):
    """Abstract base class for a single sensor's metadata reader.

    A concrete reader discovers the scene files for one sensor in a directory and
    extracts a list of sensor-agnostic *scene dicts* (see the package docstring
    for the schema) that the stereo-pair geometry code can consume without
    knowing which sensor produced them.

    Subclasses must implement :meth:`detect` (so the sensor can be chosen
    automatically) and :meth:`get_scene_dicts`.

    Attributes
    ----------
    name : str
        Human-readable sensor name (e.g. ``"WorldView"``).
    directory : str
        Path to the directory containing the sensor's metadata files.
    """

    name = "sensor"

    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str
            Path to directory containing the sensor's camera/metadata files.
        """
        self.directory = os.path.expanduser(directory)

    @classmethod
    @abstractmethod
    def detect(cls, directory, recursive=True):
        """Return True if this reader can handle the files in ``directory``.

        Parameters
        ----------
        directory : str
            Path to directory to inspect.
        recursive : bool, optional
            If True (default), also match metadata files nested in
            subdirectories. :func:`asp_plot.sensors.sensor_for_directory` first
            asks every sensor to detect shallowly and only then recursively, so
            a sensor matching at the top level wins over one matching a nested
            delivery.

        Returns
        -------
        bool
            Whether this sensor's metadata files are present.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def detect_files(cls, image_list):
        """Return True if this reader can handle the files in ``image_list``.

        The file-list counterpart of :meth:`detect`, used when the sensor must
        be chosen from an explicit list of inputs rather than a directory.

        Parameters
        ----------
        image_list : list of str
            Candidate metadata file paths.

        Returns
        -------
        bool
            Whether this sensor's metadata files are present in the list.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scene_dicts(self):
        """Return a list of per-scene metadata dictionaries.

        Returns
        -------
        list of dict
            One sensor-agnostic scene dict per scene (see package docstring).
        """
        raise NotImplementedError


def _common_base(paths):
    """Return a base directory for a list of files.

    Used to pick a working directory (for ``dg_mosaic`` outputs and pair
    naming) when a reader is built from an explicit file list rather than a
    directory. Returns the files' common parent directory, or the current
    working directory if they share no common parent or the list is empty.
    """
    paths = [os.path.abspath(p) for p in (paths or [])]
    if not paths:
        return os.getcwd()
    base = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
    # commonpath can return a file path if one entry is a prefix of another;
    # make sure we hand back a directory.
    return base if os.path.isdir(base) else os.path.dirname(base)
