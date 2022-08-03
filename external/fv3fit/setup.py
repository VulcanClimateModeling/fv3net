#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

requirements = [
    "xarray>=0.14",
    "numpy>=1.11",
    "scikit-learn>=0.22",
    "fsspec>=0.6.2",
    "pyyaml>=5.1.2",
    "tensorflow>=2.5.2",
    "typing_extensions>=3.7.4.3",
    "dacite>=1.6.0",
    "wandb[media]>=0.12.1",
    "pace-util>=0.7.0",
    "matplotlib",
    "plotly",
    "tensorboard",
    "pandas",
    "torch==1.12.0",
    "torchvision==0.13.0+cu113",
    "dgl==0.9.0",
    "geographiclib==1.52",
    "geopy==2.2.0",
]

setup_requirements = []

test_requirements = ["pytest"]

setup(
    author="The Allen Institute for Artificial Intelligence",
    author_email="jeremym@allenai.org",
    python_requires=">=3.6.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="FV3Fit is used to train machine learning models.",
    install_requires=requirements,
    dependency_links=["../loaders/", "../vcm/"],
    extras_require={},
    license="BSD license",
    long_description="FV3Fit is used to train machine learning models.",
    include_package_data=True,
    keywords="fv3fit",
    name="fv3fit",
    packages=find_packages(include=["fv3fit", "fv3fit.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ai2cm/fv3fit",
    version="0.1.0",
    zip_safe=False,
)
