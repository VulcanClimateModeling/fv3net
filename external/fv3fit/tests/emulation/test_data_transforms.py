import os
import pytest
import tempfile
import numpy as np
import tensorflow as tf
import xarray as xr
from fv3fit.emulation.data import transforms
from fv3fit.emulation.data.config import Pipeline
import vcm


@pytest.fixture
def xr_dataset():
    return xr.Dataset(
        {
            "air_temperature": xr.DataArray(
                data=np.arange(30).reshape(10, 3), dims=["sample", "z"]
            ),
            "specific_humidity": xr.DataArray(
                data=np.arange(30, 60).reshape(10, 3), dims=["sample", "z"]
            ),
        }
    )


def test_xr_dataset_to_ndarray_dataset(xr_dataset):

    result = transforms.to_ndarrays(xr_dataset)
    assert len(result) == len(xr_dataset)

    for key, data in xr_dataset.items():
        assert key in result
        result_data = result[key]
        assert isinstance(result_data, np.ndarray)
        np.testing.assert_equal(data.values, result_data)


def test_xr_dataset_to_tensor_dataset(xr_dataset):
    result = transforms.to_tensors(xr_dataset, dtype=tf.float32)
    assert len(result) == len(xr_dataset)

    for key, data in xr_dataset.items():
        assert key in result
        result_data = result[key]
        assert isinstance(result_data, tf.Tensor)
        expected = tf.convert_to_tensor(data, dtype=tf.float32)
        tf.debugging.assert_equal(result_data, expected)


@pytest.mark.parametrize(
    "lats, data, expected",
    [
        ([-55, -60, -65], [1, 2, 3], [3]),
        ([55, 60, 65], [1, 2, 3], []),
        ([-61, -65, -80], [1, 2, 3], [1, 2, 3]),
    ],
)
def test_select_antarctic(lats, data, expected):
    lats_da = xr.DataArray(np.deg2rad(lats), dims=["sample"])
    data_da = xr.DataArray(data, dims=["sample"])
    dataset = xr.Dataset({"latitude": lats_da, "field": data_da})
    result = transforms.select_antarctic(dataset)

    expected_da = xr.DataArray(expected, dims=["sample"])
    xr.testing.assert_equal(expected_da, result["field"])


def test_select_antarctic_xarray_netCDF():
    """
    xarray can't use an empty index mask (i.e., all False) along
    a selection dimension for multi-dimensional data with an unloaded
    netCDF backend
    """
    lats_da = xr.DataArray(np.deg2rad(np.linspace(40, 50, 10)), dims=["sample"])
    data_da = xr.DataArray(np.arange(20).reshape(10, 2), dims=["sample", "z"])
    dataset = xr.Dataset({"latitude": lats_da, "field": data_da})
    expected = transforms.select_antarctic(dataset)

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "saved.nc")
        dataset.to_netcdf(filename)

        loaded = xr.open_dataset(filename)
        result = transforms.select_antarctic(loaded)

    xr.testing.assert_equal(expected, result)


@pytest.mark.parametrize(
    "dataset",
    [
        xr.Dataset({"X": (["sample", "feature"], np.arange(40).reshape(10, 4))}),
        {"X": np.arange(40).reshape(10, 4)},
        {"X": tf.convert_to_tensor(np.arange(40).reshape(10, 4))},
    ],
)
def test_maybe_subselect_feature_dim_dataset_inputs(dataset):

    subselect_map = {"X": slice(2, None)}

    result = transforms.maybe_subselect_feature_dim(subselect_map, dataset)
    assert len(result["X"].shape) == len(dataset["X"].shape)
    np.testing.assert_equal(
        result["X"], np.arange(40).reshape(10, 4)[..., slice(2, None)]
    )


def test_maybe_subselect_feature_dim_empty_selection_map():

    subselect_map = {}
    full_data = np.arange(40).reshape(10, 4)

    result = transforms.maybe_subselect_feature_dim(subselect_map, {"X": full_data})
    np.testing.assert_equal(result["X"], full_data)


@pytest.mark.parametrize(
    "dataset",
    [
        {"X": np.arange(40).reshape(10, 4), "y": np.arange(20)},
        {
            "X": tf.convert_to_tensor(np.arange(40).reshape(10, 4)),
            "y": tf.convert_to_tensor(np.arange(20)),
        },
    ],
)
def test_expand_single_dim_data(dataset):

    result = transforms.expand_single_dim_data(dataset)
    assert result["X"].shape == (10, 4)
    assert result["y"].shape == (20, 1)


@pytest.mark.parametrize(
    "varname", ["air_temperature", "specific_humidity", "cloud_water_mixing_ratio"]
)
def test_derived(varname: str):

    ds = {
        f"{varname}_input": xr.DataArray(np.ones((10, 4)), dims=["sample", "feature"]),
        f"{varname}_after_precpd": xr.DataArray(
            np.ones((10, 4)) * 3, dims=["sample", "feature"]
        ),
    }

    dT_name = f"tendency_of_{varname}_due_to_microphysics"
    all_vars = list(ds.keys()) + [dT_name]

    derived = transforms.derived_dataset(all_vars, ds, tendency_timestep_sec=2)
    assert f"{varname}_input" in derived
    assert f"{varname}_after_precpd" in derived
    assert dT_name in derived
    np.testing.assert_array_equal(derived[dT_name], np.ones((10, 4)))


def test_Pipeline():
    ds = vcm.cdl_to_dataset(
        """
        netcdf Name {
            dimensions:
                sample = 1;
            variables:
                int a(sample);
            data:
                // initialize with 0
                // should end with 2
                a = 0;
        }
        """
    )

    def increment_a_ds(ds):
        return ds.assign(a=ds.a + 1)

    def increment_a_array(d):
        return {"a": d["a"] + 1}

    pipeline = Pipeline([increment_a_ds], [increment_a_array])
    out = pipeline(ds)
    assert out["a"][0].numpy() == np.array([2])
