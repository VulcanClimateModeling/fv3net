from fv3fit._shared import ArrayPacker, SliceConfig, PackerConfig
from typing import Iterable
from fv3fit._shared.packer import (
    unpack_matrix,
    pack,
    unpack,
    _unique_dim_name,
    count_features,
    clip,
)
from fv3fit.keras._models.packer import get_unpack_layer, Unpack
import pytest
import numpy as np
import xarray as xr

SAMPLE_DIM = "sample"
FEATURE_DIM = "z"

DIM_LENGTHS = {
    SAMPLE_DIM: 32,
    FEATURE_DIM: 8,
}


@pytest.fixture(
    params=["one_1d_var", "one_2d_var", "two_2d_vars", "1d_and_2d", "five_vars"]
)
def dims_list(request) -> Iterable[str]:
    if request.param == "one_1d_var":
        return [[SAMPLE_DIM]]
    elif request.param == "one_2d_var":
        return [[SAMPLE_DIM, FEATURE_DIM]]
    elif request.param == "two_2d_vars":
        return [
            [SAMPLE_DIM, FEATURE_DIM],
            [SAMPLE_DIM, FEATURE_DIM],
        ]
    elif request.param == "1d_and_2d":
        return [[SAMPLE_DIM, FEATURE_DIM], [SAMPLE_DIM]]
    elif request.param == "five_vars":
        return [
            [SAMPLE_DIM, FEATURE_DIM],
            [SAMPLE_DIM],
            [SAMPLE_DIM, FEATURE_DIM],
            [SAMPLE_DIM],
            [SAMPLE_DIM, FEATURE_DIM],
        ]


def get_array(dims: Iterable[str], value: float) -> np.ndarray:
    return np.zeros([DIM_LENGTHS[dim] for dim in dims]) + value


@pytest.fixture
def names(dims_list: Iterable[str]):
    return [f"var_{i}" for i in range(len(dims_list))]


@pytest.fixture
def dataset(names: Iterable[str], dims_list: Iterable[str]) -> xr.Dataset:
    return get_dataset(names, dims_list)


def get_dataset(names: Iterable[str], dims_list: Iterable[str]) -> xr.Dataset:
    data_vars = {}
    for i, (name, dims) in enumerate(zip(names, dims_list)):
        data_vars[name] = xr.DataArray(get_array(dims, i), dims=dims)
    ds = xr.Dataset(data_vars)
    if FEATURE_DIM in ds.dims:
        ds = ds.assign_coords({FEATURE_DIM: np.arange(DIM_LENGTHS[FEATURE_DIM])})
    return ds


@pytest.fixture
def array(names: Iterable[str], dims_list: Iterable[str]) -> np.ndarray:
    array_list = []
    for i, dims in enumerate(dims_list):
        array = get_array(dims, i)
        if len(dims) == 1:
            array = array[:, None]
        array_list.append(array)
    return np.concatenate(array_list, axis=1)


@pytest.fixture
def packer(names: Iterable[str]) -> ArrayPacker:
    return ArrayPacker(SAMPLE_DIM, names)


def test_to_array(names, dims_list, array: np.ndarray):
    dataset = get_dataset(names, dims_list)
    packer = ArrayPacker(SAMPLE_DIM, names)
    result = packer.to_array(dataset)
    np.testing.assert_array_equal(result, array)


def test_to_dataset(names, dims_list, array: np.ndarray):
    dataset = get_dataset(names, dims_list)
    packer = ArrayPacker(SAMPLE_DIM, names)
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_dataset(array)
    xr.testing.assert_equal(result, dataset)


@pytest.mark.parametrize(
    "dims_list", ["two_2d_vars", "1d_and_2d", "five_vars"], indirect=True
)
def test_get_unpack_layer(names, dims_list, array: np.ndarray):
    dataset = get_dataset(names, dims_list)
    packer = ArrayPacker(SAMPLE_DIM, names)
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = get_unpack_layer(packer, feature_dim=1)(array)
    # to_dataset does not preserve coordinates
    for name, array in zip(packer.pack_names, result):
        if array.shape[1] == 1:
            array = array[:, 0]
        np.testing.assert_array_equal(array, dataset[name])


def test_direct_unpack_layer():
    names = ["a", "b"]
    n_features = {"a": 4, "b": 4}
    packed = np.random.randn(3, 8)
    a = packed[:, :4]
    b = packed[:, 4:]
    unpack_layer = Unpack(pack_names=names, n_features=n_features, feature_dim=1)
    unpacked = unpack_layer(packed)
    np.testing.assert_array_almost_equal(unpacked[0], a)
    np.testing.assert_array_almost_equal(unpacked[1], b)


def test_unpack_before_pack_raises(names, array: np.ndarray):
    packer = ArrayPacker(SAMPLE_DIM, names)
    with pytest.raises(RuntimeError):
        packer.to_dataset(array)


def test_repack_array(names, dims_list, array: np.ndarray):
    dataset = get_dataset(names, dims_list)
    packer = ArrayPacker(SAMPLE_DIM, names)
    packer.to_array(dataset)  # must pack first to know dimension lengths
    result = packer.to_array(packer.to_dataset(array))
    np.testing.assert_array_equal(result, array)


def test_unpack_matrix():
    nz = 10

    in_ = xr.Dataset({"a": (["x", "z"], np.ones((1, nz))), "b": (["x"], np.ones((1)))})
    out = xr.Dataset(
        {"c": (["x", "z"], np.ones((1, nz))), "d": (["x", "z"], np.ones((1, nz)))}
    )

    x_packer = ArrayPacker("x", pack_names=["a", "b"])
    y_packer = ArrayPacker("x", pack_names=["c", "d"])

    in_packed = x_packer.to_array(in_)
    out_packed = y_packer.to_array(out)

    matrix = np.outer(out_packed.squeeze(), in_packed.squeeze())
    jacobian = unpack_matrix(x_packer, y_packer, matrix)

    assert isinstance(jacobian[("a", "c")], xr.DataArray)
    assert jacobian[("a", "c")].dims == ("c", "a")
    assert isinstance(jacobian[("a", "d")], xr.DataArray)
    assert jacobian[("a", "d")].dims == ("d", "a")
    assert isinstance(jacobian[("b", "c")], xr.DataArray)
    assert jacobian[("b", "c")].dims == ("c", "b")
    assert isinstance(jacobian[("b", "d")], xr.DataArray)
    assert jacobian[("b", "d")].dims == ("d", "b")


stacked_dataset = xr.Dataset({"a": xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["sample"])})


def test__unique_dim_name():
    with pytest.raises(ValueError):
        _unique_dim_name(stacked_dataset, "feature_sample")


def test_sklearn_pack(dataset: xr.Dataset, array: np.ndarray):
    packed_array, _ = pack(dataset, ["sample"])
    np.testing.assert_almost_equal(packed_array, array)


def test_sklearn_unpack(dataset: xr.Dataset):
    packed_array, feature_index = pack(dataset, ["sample"])
    unpacked_dataset = unpack(packed_array, ["sample"], feature_index)
    xr.testing.assert_allclose(unpacked_dataset, dataset)


def test_sklearn_pack_unpack_with_clipping(dataset: xr.Dataset):
    name = list(dataset.data_vars)[0]
    if FEATURE_DIM in dataset[name].dims:
        pack_config = PackerConfig({name: {FEATURE_DIM: SliceConfig(3, None)}})
        packed_array, feature_index = pack(dataset, [SAMPLE_DIM], pack_config)
        unpacked_dataset = unpack(packed_array, [SAMPLE_DIM], feature_index)
        expected = {}
        for k in dataset:
            da = dataset[k].copy(deep=True)
            indices = pack_config.clip.get(k, {})
            for dim, slice_config in indices.items():
                da = da.isel({dim: slice_config.slice})
            expected[k] = da
        xr.testing.assert_allclose(unpacked_dataset, xr.Dataset(expected))


def test_clip(dataset: xr.Dataset):
    name = list(dataset.data_vars)[0]
    if FEATURE_DIM in dataset[name].dims:
        indices = {name: {FEATURE_DIM: SliceConfig(4, 8)}}
        clipped_data = clip(dataset, indices)
        expected_da = dataset[name]
        for dim, slice_config in indices[name].items():
            expected_da = expected_da.isel({dim: slice_config.slice})
        xr.testing.assert_identical(clipped_data[name], expected_da)


def test_count_features():
    SAMPLE_DIM_NAME = "axy"
    ds = xr.Dataset(
        data_vars={
            "a": xr.DataArray(np.zeros([10]), dims=[SAMPLE_DIM_NAME]),
            "b": xr.DataArray(np.zeros([10, 1]), dims=[SAMPLE_DIM_NAME, "b_dim"]),
            "c": xr.DataArray(np.zeros([10, 5]), dims=[SAMPLE_DIM_NAME, "c_dim"]),
        }
    )
    names = list(ds.data_vars.keys())
    assert len(names) == 3
    index = ds.to_stacked_array(
        "feature", [SAMPLE_DIM_NAME], variable_dim="var"
    ).indexes["feature"]
    out = count_features(index, variable_dim="var")
    assert len(out) == len(names)
    for name in names:
        assert name in out
    assert out["a"] == 1
    assert out["b"] == 1
    assert out["c"] == 5


def test_array_packer_dump_and_load(tmpdir, dataset):
    packer = ArrayPacker(SAMPLE_DIM, list(dataset.data_vars))
    packer.to_array(dataset)
    with open(str(tmpdir.join("packer.yaml")), "w") as f:
        packer.dump(f)
    with open(str(tmpdir.join("packer.yaml"))) as f:
        loaded_packer = ArrayPacker.load(f)
    packer._pack_names = loaded_packer._pack_names
    packer._n_features = loaded_packer._n_features
    packer._sample_dim_name = loaded_packer._sample_dim_name
    packer._feature_index = loaded_packer._feature_index


@pytest.mark.skip
@pytest.mark.parametrize(
    "feature_dims",
    [
        pytest.param([("var1", {"z": 8})], id="one_var_one_feature_dim",),
        pytest.param([("var1", {"z": 8, "y": 5})], id="one_var_two_feature_dims",),
        pytest.param(
            [("var1", {"z": 8, "y": 5}), ("var2", {"y": 5})],
            id="two_vars_two_feature_dims",
        ),
        pytest.param(
            [("var1", {"z": 8}), ("var2", {"z": 8})], id="two_vars_one_feature_dim",
        ),
        pytest.param(
            [("var1", {"z": 8}), ("var2", {"z": 5})],
            id="two_vars_diff_sized_feature_dim",
        ),
        pytest.param(
            [("var1", {"z1": 8}), ("var2", {"z2": 5})],
            id="two_vars_diff_two_feature_dims",
        ),
    ],
)
def test_to_stacked_array(feature_dims):
    dataset_as_dict = {}
    for name, dims in feature_dims:
        full_dims = [SAMPLE_DIM] + list(dims)
        full_dim_sizes = [10] + list(dims.values())
        dataset_as_dict[name] = xr.DataArray(np.zeros(full_dim_sizes), dims=full_dims)
    dataarray = to_stacked_array(dataset_as_dict, "NEW_DIM", [SAMPLE_DIM])
    # dataarray = xr.Dataset(dataset_as_dict).to_stacked_array("NEW_DIM", [SAMPLE_DIM])
    # to form xarray dataset, need coords for dims of different sizes
    dataset = {}
    for name, da in dataset_as_dict.items():
        for dim in da.dims:
            if dim not in da.coords:
                dataset[name] = da.assign_coords({dim: list(range(da.sizes[dim]))})
            else:
                dataset[name] = da
    dataset = xr.Dataset(dataset)
    # assert dataarray.shape == expected_array_shape
    print(dataarray)
    print(dataarray.sel({"variable": "var1"}))
    # assert 0 == 1

    xr.testing.assert_identical(dataset, dataarray.to_unstacked_dataset("NEW_DIM"))
