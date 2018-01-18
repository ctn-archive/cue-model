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
        'nengo[all_solvers]',
        'nengo_extras',
        'nengo_spa==0.3',
        'numpy',
        'pandas',
        'pytry',
        'scipy',
        'seaborn',
        'statsmodels',
    ],
    dependency_links=[
        'git+ssh://git@github.com/nengo/nengo_extras.git'
        '@d63e12aa787419fcafed32027105583d614e9e6d#egg=nengo-extras-0.1.0.dev0'
    ]
)
