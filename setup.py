#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="asp_plot",
    version="0.1.1",
    description="Python library and client for plotting Ames Stereo Pipeline outputs",
    author="Ben Purinton",
    author_email="purinton@uw.edu",
    packages=find_packages(),
    entry_points={"console_scripts": ["asp_plot=asp_plot.cli.asp_plot:main"]},
)
