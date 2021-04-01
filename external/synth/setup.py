# -*- coding: utf-8 -*-
from setuptools import setup


setup(
    long_description="",
    name="synth",
    version="0.1.0",
    python_requires="==3.*,>=3.7.0",
    author="Noah D. Brenowitz",
    author_email="noahb@vulcan.com",
    packages=["synth"],
    package_dir={"": "."},
    package_data={"synth": ["_dataset_fixtures/*.json"]},
    entry_points={"console_scripts": ["synth-read-schema=synth.clis:read_schema"]},
    install_requires=[
        "dask==2.*,>=2.15.0",
        "fsspec==0.*,>=0.7.3",
        "toolz==0.*,>=0.10.0",
        "xarray==0.*,>=0.15.1",
        "zarr==2.*,>=2.4.0",
    ],
    extras_require={"dev": ["pytest==5.*,>=5.2.0", "pytest-regtest==1.*,>=1.4.4"]},
)
