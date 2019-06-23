#!/usr/bin/env python

from setuptools import find_packages, setup


setup(
    name="cue",
    version='1.0-dev',
    author="Jan Gosmann",
    author_email="jgosmann@uwaterloo.ca",

    packages=find_packages(),
    provides=['cue'],
    install_requires=[
        'matplotlib',
        'nengo[optional]>=2.8,<2.9',
        'nengo_extras',
        'nengo_spa==0.3',
        'numpy',
        'pandas',
        'psyrun',
        'pytry',
        'scipy',
        'seaborn',
        'statsmodels',
    ],
)
