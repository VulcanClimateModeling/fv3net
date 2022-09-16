import numpy as np
import pytest
import requests
import xarray as xr
import unittest.mock
import tempfile

np.random.seed(0)


@pytest.fixture(scope="session")
def state(tmp_path_factory):
    url = "https://github.com/ai2cm/vcm-ml-example-data/blob/b100177accfcdebff2546a396d2811e32c01c429/fv3net/prognostic_run/inputs_4x4.nc?raw=true"  # noqa
    r = requests.get(url)
    lpath = tmp_path_factory.getbasetemp() / "input_data.nc"
    lpath.write_bytes(r.content)
    return xr.open_dataset(str(lpath))


@pytest.fixture(scope="session", autouse=True)
def emulation_cache_tmpdir():
    with tempfile.TemporaryDirectory() as tmpdir:
        with unittest.mock.patch("fv3fit.data.netcdf.io.CACHE_DIR", tmpdir):
            yield
