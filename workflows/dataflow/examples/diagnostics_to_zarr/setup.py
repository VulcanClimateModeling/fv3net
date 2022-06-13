from setuptools import find_packages, setup

dependencies = [
    "apache-beam",
    "cloudpickle",
    "dask",
    "gcsfs",
    "fsspec",
    "google-cloud-storage",
    "intake",
    "numba",
    "netCDF4",
    "xarray==0.15.0",
    "partd",
    "pyyaml>=5.0",
    "xgcm",
    "zarr",
]

setup(
    name="diags-to-zarr",
    packages=find_packages(),
    install_requires=dependencies,
    version="0.1.0",
    description="Improving the GFDL FV3 model physics with machine learning",
    author="The Allen Institute for Artificial Intelligence",
    license="MIT",
)
