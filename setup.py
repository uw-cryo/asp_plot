#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="asp_plot",
    version="0.5.1",
    description="Python library and client for plotting Ames Stereo Pipeline outputs",
    author="Ben Purinton",
    author_email="purinton@uw.edu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "asp_plot=asp_plot.cli.asp_plot:main",
            "camera_optimization=asp_plot.cli.camera_optimization:main",
        ]
    },
)
