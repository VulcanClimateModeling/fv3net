import os
import pytest
import synth
import xarray as xr
import numpy as np
import loaders
import loaders.mappers
import cftime
from loaders.batches._batch import (
    batches_from_mapper,
    _get_batch,
)

DATA_VARS = ["air_temperature", "specific_humidity"]
Z_DIM_SIZE = 79


class MockDatasetMapper:
    def __init__(self, schema: synth.DatasetSchema):
        self._schema = schema
        self._keys = [f"2000050{i+1}.000000" for i in range(4)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = synth.generate(self._schema).drop("initial_time")
        return ds

    def keys(self):
        return self._keys

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())


@pytest.fixture(params=["MockDatasetMapper", "MultiDatasetMapper"])
def mapper(request, datadir):
    one_step_zarr_schema = "one_step_zarr_schema.json"
    # uses the one step schema but final mapper
    # functions the same for all data sources
    with open(os.path.join(datadir, one_step_zarr_schema)) as f:
        schema = synth.load(f)
    mapper = MockDatasetMapper(schema)
    if request.param == "MockDatasetMapper":
        return mapper
    elif request.param == "MultiDatasetMapper":
        return loaders.mappers.MultiDatasetMapper([mapper, mapper, mapper])
    else:
        raise ValueError("Invalid mapper type provided.")


@pytest.fixture
def random_state():
    return np.random.RandomState(0)


def test__get_batch(mapper):
    ds = _get_batch(
        mapper=mapper,
        data_vars=["air_temperature", "specific_humidity"],
        keys=mapper.keys(),
    )
    assert len(ds["time"]) == 4


def test_batches_from_mapper(mapper):
    batched_data_sequence = batches_from_mapper(
        mapper, DATA_VARS, timesteps_per_batch=2, needs_grid=False,
    )
    assert len(batched_data_sequence) == 2
    expected_num_samples = 6 * 48 * 48 * 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)
        for name in batch.data_vars.keys():
            assert batch[name].dims[0] == loaders.SAMPLE_DIM_NAME
            assert batch[name].sizes[loaders.SAMPLE_DIM_NAME] == expected_num_samples


@pytest.mark.parametrize(
    "total_times,times_per_batch,valid_num_batches", [(3, 1, 3), (3, 2, 2), (3, 4, 1)]
)
def test_batches_from_mapper_timestep_list(
    mapper, total_times, times_per_batch, valid_num_batches
):
    timestep_list = list(mapper.keys())[:total_times]
    batched_data_sequence = batches_from_mapper(
        mapper,
        DATA_VARS,
        timesteps_per_batch=times_per_batch,
        timesteps=timestep_list,
        needs_grid=False,
    )
    print(batched_data_sequence._args)
    assert len(batched_data_sequence) == valid_num_batches
    timesteps_used = sum(batched_data_sequence._args, ())  # flattens list
    assert set(timesteps_used).issubset(timestep_list)


def test__batches_from_mapper_invalid_times(mapper):
    invalid_times = list(mapper.keys())[:2] + ["20000101.000000", "20000102.000000"]
    with pytest.raises(ValueError):
        batches_from_mapper(
            mapper,
            DATA_VARS,
            timesteps_per_batch=2,
            timesteps=invalid_times,
            needs_grid=False,
        )


def test_diagnostic_batches_from_mapper(mapper):
    batched_data_sequence = batches_from_mapper(
        mapper, DATA_VARS, timesteps_per_batch=2, training=False, needs_grid=False,
    )
    assert len(batched_data_sequence) == len(mapper) // 2 + len(mapper) % 2
    for i, batch in enumerate(batched_data_sequence):
        assert len(batch["z"]) == Z_DIM_SIZE
        assert set(batch.data_vars) == set(DATA_VARS)


@pytest.mark.parametrize(
    "tiles",
    [
        pytest.param([1, 2, 3, 4, 5, 6], id="one-indexed"),
        pytest.param([0, 1, 2, 3, 4, 5], id="zero-indexed"),
    ],
)
def test_baches_from_mappper_different_indexing_conventions(tiles):
    n = 48
    ds = xr.Dataset(
        {"a": (["time", "tile", "y", "x"], np.zeros((1, 6, n, n)))},
        coords={"time": [cftime.DatetimeJulian(2016, 8, 1)], "tile": tiles},
    )
    mapper = loaders.mappers.XarrayMapper(ds)
    seq = batches_from_mapper(mapper, ["a", "lon"], res=f"c{n}")

    assert len(seq) == 1
    assert ds.a[0].size == seq[0].a.size
