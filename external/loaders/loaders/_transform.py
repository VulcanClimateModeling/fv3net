import numpy as np
from numpy.random import RandomState
from typing import Mapping, Tuple, Sequence, Union
import xarray as xr
from toolz import groupby
from datetime import timedelta

from vcm import safe, parse_datetime_from_str

from .constants import SAMPLE_DIM_NAME

Z_DIM_NAMES = ["z", "pfull"]

Time = str
Tile = int
K = Tuple[Time, Tile]


def stack_dropnan_shuffle(
    init_time_dim_name: str, random_state: RandomState, ds: xr.Dataset,
) -> xr.Dataset:
    ds = ds.load()
    stack_dims = [dim for dim in ds.dims if dim not in Z_DIM_NAMES]
    if len(set(ds.dims).intersection(Z_DIM_NAMES)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    ds_stacked = safe.stack_once(
        ds,
        SAMPLE_DIM_NAME,
        stack_dims,
        allowed_broadcast_dims=Z_DIM_NAMES + [init_time_dim_name],
    )
    ds_no_nan = ds_stacked.dropna(SAMPLE_DIM_NAME)
    if len(ds_no_nan[SAMPLE_DIM_NAME]) == 0:
        raise ValueError(
            "No Valid samples detected. Check for errors in the training data."
        )
    ds = ds_no_nan.load()
    return shuffled(ds, SAMPLE_DIM_NAME, random_state)


def shuffled(
    dataset: xr.Dataset, dim: str, random: np.random.RandomState
) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dataset: input data to be shuffled
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
    """
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({dim: shuffled_inds})


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


class GroupByTime:
    def __init__(self, tiles: Mapping[K, xr.Dataset]) -> Mapping[K, xr.Dataset]:
        def fn(key):
            time, _ = key
            return time

        self._tiles = tiles
        self._time_lookup = groupby(fn, self._tiles.keys())

    def keys(self):
        return self._time_lookup.keys()

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, time: Time) -> xr.Dataset:
        datasets = [self._tiles[key] for key in self._time_lookup[time]]
        tiles = range(len(datasets))
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)
