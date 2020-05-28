import pytest
import os
import xarray as xr
import numpy as np
from distutils import dir_util

import synth

from fv3net.regression.loaders._nudged import (
    _get_batch_slices,
    load_nudging_batches,
    _get_path_for_nudging_timescale,
)


@pytest.fixture
def xr_dataset():

    dat = np.arange(120).reshape(5, 2, 3, 4)
    ds = xr.Dataset(
        data_vars={"data": (("time", "x", "y", "z"), dat)},
        coords={
            "time": np.arange(5),
            "x": np.arange(2),
            "y": np.arange(3),
            "z": np.arange(4),
        },
    )

    return ds


@pytest.mark.regression
def test_load_nudging_batches(datadir):

    xlim = 10
    ylim = 10
    zlim = 2
    tlim = 144
    ntimes = 90
    init_time_skip_hr = 12  # 48 15-min timesteps
    num_batches = 14

    rename = {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
    }
    input_vars = ["air_temperature", "specific_humidity"]
    output_vars = ["dQ1", "dQ2"]
    data_vars = input_vars + output_vars
    local_vars_for_storag = input_vars + list(rename.keys())

    synth_data = ["nudging_tendencies", "after_physics"]
    for key in synth_data:
        schema_path = datadir.join(f"{key}.json")
        # output directory with nudging timescale expected
        zarr_out = datadir.join(f"outdir-3h/{key}.zarr")

        with open(str(schema_path)) as f:
            schema = synth.load(f)

        xr_zarr = synth.generate(schema)
        reduced_ds = xr_zarr[[var for var in local_vars_for_storag if var in xr_zarr]]
        decoded = xr.decode_cf(reduced_ds)
        # limit data for efficiency (144 x 6 x 2 x 10 x 10)
        decoded = decoded.isel(
            time=slice(0, tlim), x=slice(0, xlim), y=slice(0, ylim), z=slice(0, zlim)
        )
        decoded.to_zarr(str(zarr_out))

    # skips first 48 timesteps, only use 90 timesteps
    sequence = load_nudging_batches(
        str(datadir),
        data_vars,
        timescale_hours=3,
        num_batches=num_batches,
        rename_variables=rename,
        initial_time_skip_hr=init_time_skip_hr,
        n_times=ntimes,
    )

    # 14 batches requested
    assert len(sequence._args) == num_batches

    batch_samples_total = 0
    for batch in sequence:
        batch_samples_total += batch.sizes["sample"]

    total_samples = ntimes * 6 * xlim * ylim
    expected_num_samples = (total_samples // num_batches) * num_batches
    assert batch_samples_total == expected_num_samples


@pytest.mark.parametrize(
    "num_samples,samples_per_batch,num_batches", [(5, 2, None), (5, 4, 2)]
)
def test__get_batch_slices(num_samples, samples_per_batch, num_batches):

    expected = [slice(0, 2), slice(2, 4)]
    args = _get_batch_slices(num_samples, samples_per_batch, num_batches=num_batches)
    assert args == expected


@pytest.mark.parametrize(
    "num_samples,samples_per_batch,num_batches", [(5, 6, None), (5, 2, 6)]
)
def test__get_batch_slices_failure(num_samples, samples_per_batch, num_batches):
    with pytest.raises(ValueError):
        _get_batch_slices(num_samples, samples_per_batch, num_batches=num_batches)


@pytest.fixture
def nudging_output_dirs(tmpdir):

    # nudging dirs which might be confusing for parser
    dirs = ["1.00", "1.5", "15"]

    nudging_dirs = {}
    for item in dirs:
        curr_dir = os.path.join(tmpdir, f"outdir-{item}h")
        os.mkdir(curr_dir)
        nudging_dirs[item] = curr_dir

    return nudging_dirs


@pytest.mark.parametrize(
    "timescale, expected_key",
    [(1, "1.00"), (1.0, "1.00"), (1.5, "1.5"), (1.500001, "1.5")],
)
def test__get_path_for_nudging_timescale(nudging_output_dirs, timescale, expected_key):

    expected_path = nudging_output_dirs[expected_key]
    result_path = _get_path_for_nudging_timescale(
        nudging_output_dirs.values(), timescale, tol=1e-5
    )
    assert result_path == expected_path


@pytest.mark.parametrize("timescale", [1.1, 1.00001])
def test__get_path_for_nudging_timescale_failure(nudging_output_dirs, timescale):
    with pytest.raises(KeyError):
        _get_path_for_nudging_timescale(nudging_output_dirs, timescale, tol=1e-5)
