import pickle

import numpy as np
import zarr

import pytest
import xarray as xr

from synth.core import _Encoder, load_directory_schema

from synth import (
    generate,
    read_schema_from_zarr,
    read_directory_schema,
    dump_directory_schema_to_disk,
    Array,
    ChunkedArray,
    CoordinateSchema,
    DatasetSchema,
    Range,
    VariableSchema,
    __version__,
    dumps,
    load,
)

from synth.core import dict_to_schema


def test_version():
    assert __version__ == "0.1.0"


def test_generate_array():
    r = Range(0, 1)
    a = Array(shape=(1, 2), dtype=np.float32)
    out = a.generate(r)

    assert out.dtype == a.dtype
    assert out.shape == a.shape


def test_generate_chunked_array():
    r = Range(0, 1)
    a = ChunkedArray(shape=(10, 10), dtype=np.float32, chunks=((5, 5), (5, 5)))
    out = a.generate(r)

    assert out.dtype == a.dtype
    assert out.shape == a.shape
    assert out.chunks == a.chunks


coord1 = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
coord2 = CoordinateSchema("x", ["x"], np.array([1, 2, 3]), attrs={"attr1": "something"})
coord3 = CoordinateSchema("y", ["x"], np.array([1, 2, 3]))
coord4 = CoordinateSchema("x", ["y"], np.array([1, 2, 3]))


@pytest.mark.parametrize(
    "coordA,coordB", [(coord1, coord1), (coord1, coord2)],
)
def test_coord_schema_equivalence(coordA, coordB):
    assert coordA == coordB


@pytest.mark.parametrize(
    "coordA,coordB", [(coord1, coord3), (coord1, coord4)],
)
def test_coord_schema_not_equivalent(coordA, coordB):
    assert coordA != coordB


variable1 = VariableSchema(
    "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32"))
)
variable2 = VariableSchema(
    "a",
    ["x"],
    ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
    attrs={"attr1": "something"},
)
variable3 = VariableSchema(
    "a", ["x"], ChunkedArray(shape=[4], chunks=[2], dtype=np.dtype("float64"))
)
variable4 = VariableSchema(
    "a",
    ["x", "y"],
    ChunkedArray(shape=[3, 2], chunks=[1, 1], dtype=np.dtype("float32")),
)
variable5 = VariableSchema(
    "b", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32"))
)


@pytest.mark.parametrize(
    "variableA,variableB",
    [(variable1, variable1), (variable1, variable2), (variable1, variable3)],
)
def test_variable_schema_equivalence(variableA, variableB):
    assert variableA == variableB


@pytest.mark.parametrize(
    "variableA,variableB", [(variable1, variable4), (variable1, variable5)],
)
def test_variable_schema_not_equivalent(variableA, variableB):
    assert variableA != variableB


dataset1 = DatasetSchema({"x": coord1}, {"a": variable1})
dataset2 = DatasetSchema({"x": coord2}, {"a": variable2})
dataset3 = DatasetSchema({"y": coord1}, {"a": variable4})
dataset4 = DatasetSchema({"x": coord1}, {"a": variable1, "b": variable5})


@pytest.mark.parametrize(
    "datasetA,datasetB,expected",
    [
        (dataset1, dataset1, True),
        (dataset1, dataset2, True),
        (dataset1, dataset3, False),
        (dataset1, dataset4, False),
    ],
)
def test_dataset_schema_equivalence(datasetA, datasetB, expected):
    assert (datasetA == datasetB) == expected


def test_encoder_dtype():
    e = _Encoder()
    assert e.encode(np.dtype("float32")) == '"' + np.dtype("float32").str + '"'


def test_encoder_array():
    e = _Encoder()
    assert e.encode(np.array([1, 2, 3])) == "[1, 2, 3]"


def test_DatasetSchema_dumps():

    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
    )
    ds = DatasetSchema(coords=[x], variables=[a])

    val = dumps(ds)
    assert isinstance(val, str)


def test_DatasetSchema_dumps_regression(regtest):
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
    )

    ds = DatasetSchema(coords={x.name: x}, variables={a.name: a})

    val = dumps(ds)
    print(val, file=regtest)


mock_dict_schema_v1 = {
    "coords": [{"name": "x", "dims": ["x"], "value": [1, 2, 3], "attrs": [1]}],
    "variables": [
        {
            "name": "a",
            "dims": ["x"],
            "array": {"shape": [3], "dtype": "<f4", "chunks": [1]},
            "range": {"min": 0, "max": 10},
        }
    ],
}

mock_dict_schema_v2alpha = {
    "version": "v2alpha",
    "schema": {
        "coords": [{"name": "x", "dims": ["x"], "value": [1, 2, 3], "attrs": [1]}],
        "variables": [
            {
                "name": "a",
                "dims": ["x"],
                "array": {"shape": [3], "dtype": "<f4", "chunks": [1]},
            }
        ],
    },
}

mock_dict_schema_v3 = {
    "version": "v3",
    "schema": {
        "coords": {"x": {"name": "x", "dims": ["x"], "value": [1, 2, 3], "attrs": [1]}},
        "variables": {
            "a": {
                "name": "a",
                "dims": ["x"],
                "array": {"shape": [3], "dtype": "<f4", "chunks": [1]},
            }
        },
    },
}


@pytest.mark.parametrize(
    "d", [mock_dict_schema_v1, mock_dict_schema_v2alpha, mock_dict_schema_v3]
)
def test_DatasetSchemaLoads(d):

    ds = dict_to_schema(d)
    assert isinstance(ds, DatasetSchema)

    v = ds.variables["a"]
    assert isinstance(v, VariableSchema)

    c = ds.coords["x"]
    assert c.attrs == [1]


def test_generate_and_pickle_integration():
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32"))
    )

    ranges = {"a": Range(0, 10)}

    ds = DatasetSchema(coords={x.name: x}, variables={a.name: a})
    d = generate(ds, ranges)
    pickle.dumps(d)


def test_generate_regression(regtest):
    """
    Note:

        regtest fixture provided by pytest-regtest plugin:
        https://pypi.org/project/pytest-regtest/

    """
    x = CoordinateSchema("x", ["x"], np.array([1, 2, 3]))
    a = VariableSchema(
        "a", ["x"], ChunkedArray(shape=[3], chunks=[1], dtype=np.dtype("float32")),
    )

    ranges = {"a": Range(0, 10)}

    ds = DatasetSchema(coords={x.name: x}, variables={a.name: a})
    d = generate(ds, ranges)
    arr = d.a.values
    print(arr, file=regtest)


def test_cftime_generate():
    julian_time_attrs = {
        "calendar": "julian",
        "calendar_type": "JULIAN",
        "cartesian_axis": "T",
        "long_name": "time",
        "units": "seconds since 2016-08-01T00:15:00.000026",
    }

    store = {}

    group = zarr.open_group(store, mode="w")
    arr = group.zeros("time", shape=[1], dtype=np.float64)
    arr.attrs.update(julian_time_attrs)
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]

    schema = read_schema_from_zarr(group, coords=["time"])
    ds = generate(schema, {})

    assert dict(ds.time.attrs) == dict(julian_time_attrs)


def test_generate_coord():
    s = CoordinateSchema(name="zaxis_1", dims=["zaxis_1"], value=[1.0],)
    expected = xr.DataArray(name="zaxis_1", dims=["zaxis_1"], data=[1.0])
    xr.testing.assert_equal(generate(s), expected)


def test_directory_schema_integrations(tmpdir):
    ds = xr.DataArray(name="zaxis_1", dims=["zaxis_1"], data=[1.0]).to_dataset(name="a")
    ds.to_zarr(str(tmpdir.join("a.zarr")))

    schema = read_directory_schema(str(tmpdir))
    assert "a.zarr" in schema

    schema_dir = tmpdir.join("schema")
    dump_directory_schema_to_disk(schema, str(schema_dir))

    # now try loading the schema
    loaded = load_directory_schema(str(schema_dir))
    assert loaded == schema


def test_dump_directory_schema(tmpdir):
    dump_directory_schema_to_disk({"a.zarr": dataset1}, str(tmpdir))
    # test that we can load schema from within the saved directory
    with open(str(tmpdir.join("a.zarr.json"))) as f:
        load(f)
