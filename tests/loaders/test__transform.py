from fv3net.regression.loaders._transform import _shuffled, _chunk_indices
import numpy as np
import xarray as xr


def test__chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = _chunk_indices(chunks)
    assert ans == expected


def _dataset(sample_dim):
    m, n = 10, 2
    x = "x"
    sample = sample_dim
    return xr.Dataset(
        {"a": ([sample, x], np.ones((m, n))), "b": ([sample], np.ones((m)))},
        coords={x: np.arange(n), sample_dim: np.arange(m)},
    )


def test__shuffled():
    dataset = _dataset("sample")
    dataset.isel(sample=1)
    _shuffled(dataset, "sample", np.random.RandomState(1))


def test__shuffled_dask():
    dataset = _dataset("sample").chunk()
    _shuffled(dataset, "sample", np.random.RandomState(1))
