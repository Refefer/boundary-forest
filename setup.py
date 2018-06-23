#!/usr/bin/env python
import os

from setuptools import setup, find_packages
from distutils.core import setup

BASE = os.path.dirname(os.path.abspath(__file__))

setup(
    name='boundary-forest',
    version="0.0.1",
    description="Boundary Forest Implementation",
    url="https://www.github.com/Refefer/boundary-forest",
    packages=find_packages(BASE),
    scripts=[],
    install_requires=[
        # See requirements.txt
    ],
    author='Andrew Stanton')
