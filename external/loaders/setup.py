from setuptools import setup, find_packages


setup(
    name="loaders",
    version="0.1.0",
    python_requires=">=3.6.9",
    author="Anna Kwa",
    author_email="annak@allenai.org",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={},
    entry_points={
        "console_scripts": [
            "validate_batches_config=loaders.validate_batches_config:main",
        ]
    },
    install_requires=[
        "fsspec>=0.7.4",
        "numpy>=1.18.4",
        "pyyaml>=5.3.0",
        "toolz>=0.10.0",
        "xarray>=0.15.1",
        "zarr>=2.4.0",
        "joblib>=0.16.0",
        # cannot refer to vcm packages here...only pypi
        # "vcm",
        "m2r2",
    ],
    dependency_links=["../vcm"],
)
