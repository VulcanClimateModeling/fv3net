"""
Concepts:

1. sampler (needs to be efficient)
    Some dimensions have NaNs (e.g. surface fields with step='begin' step='output')
2. random variable (trained from sample)
    Single variate or multivariate (along a height dimension for instance)
3. output-description


chunks: (we need to be to generate chunked data)


"""
from typing import Sequence, Tuple, Mapping
import json
from dataclasses import dataclass, asdict, field
import zarr
import numpy as np
import dask.array as da
import xarray as xr
import logging
import io
import cftime
import datetime
from functools import singledispatch

from toolz import compose
import os
import fsspec

logger = logging.getLogger(__file__)

SCHEMA_VERSION = "v3"


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (cftime.DatetimeJulian, datetime.datetime)):
            return o.isoformat()
        try:
            return o.str
        except AttributeError:
            pass
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, o)


@dataclass
class Range:
    min: float
    max: float

    def _generate_like(self, x):
        return self.generate_array(x.shape, x.dtype)

    def generate_chunked(self, shape, chunks, dtype) -> da.Array:
        darr = da.empty(shape, chunks=chunks, dtype=dtype)
        return darr.map_blocks(self._generate_like)

    def generate_array(self, shape, dtype) -> np.ndarray:
        # need to set random seed here
        # to make this generation deterministic
        np.random.seed(0)
        return np.random.uniform(low=self.min, high=self.max, size=shape).astype(dtype)


@dataclass
class Array:
    shape: Sequence[int]
    dtype: np.dtype

    def generate(self, range: Range):
        return range.generate_array(self.shape, self.dtype)


@dataclass
class ChunkedArray:
    shape: Sequence[int]
    dtype: np.dtype
    chunks: Tuple[Tuple[int]]

    # TODO probably remove these generating functions
    # seems a poor separation of concerns.
    def generate(self, range: Range):
        return range.generate_chunked(self.shape, self.chunks, self.dtype)


@dataclass
class VariableSchema:
    name: str = field(compare=True)
    dims: Sequence[str] = field(compare=True)
    array: ChunkedArray = field(compare=False)
    attrs: Mapping = field(default_factory=dict, compare=False)


@dataclass
class CoordinateSchema:
    name: str = field(compare=True)
    dims: Sequence[str] = field(compare=True)
    value: np.ndarray = field(compare=False)
    attrs: Mapping = field(default_factory=dict, compare=False)


@dataclass
class DatasetSchema:
    coords: Mapping[str, CoordinateSchema]
    variables: Mapping[str, VariableSchema]


@singledispatch
def generate(_):
    pass


@generate.register
def _(self: CoordinateSchema):
    return xr.DataArray(self.value, dims=self.dims, name=self.name, attrs=self.attrs)


@generate.register
def _(self: VariableSchema, range: Range):
    return xr.DataArray(
        self.array.generate(range), dims=self.dims, name=self.name, attrs=self.attrs,
    )


@generate.register
def _(self: DatasetSchema, ranges: Mapping[str, Range] = None):
    ranges = {} if ranges is None else ranges
    default_range = Range(-1000, 1000)
    return xr.Dataset(
        {
            variable: generate(schema, ranges.get(variable, default_range))
            for variable, schema in self.variables.items()
        },
        coords={coord: generate(schema) for coord, schema in self.coords.items()},
    )


# TODO test this function
def read_schema_from_zarr(
    group: zarr.Group,
    coords=("forecast_time", "time", "initial_time", "tile", "step", "z", "y", "x"),
) -> DatasetSchema:

    variables = {}
    coord_schemes = {}

    for variable in group:
        logger.info(f"Reading {variable}")
        arr = group[variable]
        attrs = dict(arr.attrs)

        dims = attrs.pop("_ARRAY_DIMENSIONS")

        if variable in coords:
            scheme = CoordinateSchema(variable, [variable], arr[:], attrs)
            coord_schemes[variable] = scheme
        else:
            array = ChunkedArray(arr.shape, arr.dtype, arr.chunks)
            scheme = VariableSchema(variable, dims, array, attrs=attrs)
            variables[variable] = scheme

    return DatasetSchema(coord_schemes, variables)


def read_directory_schema(rundir: str) -> Mapping[str, DatasetSchema]:
    """Read schema from a directories

    Does not support recursive directories or files other than zarr

    See Also:

        dump_directory_schema_to_disk

    """
    import fsspec

    fs, _, (path_with_no_prefix,) = fsspec.get_fs_token_paths(rundir)
    files = fs.ls(path_with_no_prefix)

    read_schema = compose(read_schema_from_zarr, zarr.open_group, fs.get_mapper)

    return {
        os.path.relpath(file, path_with_no_prefix): read_schema(file)
        for file in files
        if file.endswith(".zarr")
    }


def _write_dataset(ds: xr.Dataset, outpath):
    _, ext = os.path.splitext(outpath)
    if ext == '.zarr':
        ds.to_zarr(outpath, consolidated=True)
    elif ext =='.nc':
        ds.to_netcdf(outpath, engine='h5netcdf')
    else:
        raise NotImplementedError("Extension {ext} is not supported.")


def write_directory_schema(
    directory: str, schema: Mapping[str, DatasetSchema], ranges=None
):
    for relpath, schema in schema.items():

        ds = generate(schema, ranges)
        outpath = os.path.join(directory, relpath)
        _write_dataset(ds, outpath)


def dump_directory_schema_to_disk(zarrs: Mapping[str, DatasetSchema], path: str):
    """Dumps the schema for a mapping from strings to DatasetSchema to a
    directory

    Example:

        >>> import synth
        >>> url = "gs://vcm-ml-scratch/test-end-to-end-integration/integration-test-cf6e957cea82/nudge_to_fine_run/" # noqa
        >>> schema = synth.read_directory_schema(url)
        >>> synth.dump_directory_schema_to_disk(schema, "nudge_to_fine_schema")
        >>> import os
        >>> import pprint
        >>> pprint.pprint(os.listdir("nudge_to_fine_schema"))
        ['atmos_dt_atmos.zarr.json',
        'physics_tendencies.zarr.json',
        'diags.zarr.json',
        'state_after_timestep.zarr.json',
        'nudging_tendencies.zarr.json',
        'sfc_dt_atmos.zarr.json']

    """
    mapper = fsspec.get_mapper(path)
    for rel_path, schema in zarrs.items():
        mapper[rel_path + ".json"] = dumps(schema).encode()


def load_directory_schema(path: str) -> DatasetSchema:
    """Load the schema contained in a local directory"""
    files = os.listdir(path)

    out = {}
    for file in files:
        base, ext = os.path.splitext(file)
        abspath = os.path.join(path, file)
        if ext == ".json":
            with open(abspath) as f:
                out[base] = load(f)
    return out


def read_schema_from_dataset(dataset: xr.Dataset) -> DatasetSchema:

    variables = {}
    coord_schemes = {}

    for coord in dataset.coords:
        logger.info(f"Reading coordinate {coord}")
        arr = dataset[coord].values
        attrs = dict(dataset[coord].attrs)
        dims = [dim for dim in dataset[coord].dims]
        scheme = CoordinateSchema(coord, dims, arr, attrs)
        coord_schemes[coord] = scheme

    for variable in dataset:
        logger.info(f"Reading {variable}")
        arr = dataset[variable].data
        chunks = dataset[variable].chunks
        if chunks is None:
            chunks = dataset[variable].values.shape
        attrs = dict(dataset[variable].attrs)
        dims = [dim for dim in dataset[variable].dims]
        array = ChunkedArray(arr.shape, arr.dtype, chunks)
        scheme = VariableSchema(variable, dims, array, attrs=attrs)
        variables[variable] = scheme

    return DatasetSchema(coord_schemes, variables)


def dump(schema: DatasetSchema, fp):
    output = {"version": SCHEMA_VERSION, "schema": asdict(schema)}
    json.dump(output, fp, cls=_Encoder, indent=4)


def dumps(schema: DatasetSchema):
    fp = io.StringIO()
    dump(schema, fp)
    return fp.getvalue()


# To bump the version make a new parser named
# dict_to_schema_<version>
def dict_to_schema_v1_v2(d):

    coords = {}
    for coord in d["coords"]:
        coords[coord["name"]] = CoordinateSchema(**coord)

    variables = {}
    for variable in d["variables"]:
        array_spec = variable.pop("array")
        array_spec.update(
            {
                "shape": tuple(array_spec["shape"]),
                "dtype": np.dtype(array_spec["dtype"]),
                "chunks": tuple(array_spec["chunks"]),
            }
        )
        array = ChunkedArray(**array_spec)
        variables[variable["name"]] = VariableSchema(
            array=array,
            name=variable["name"],
            dims=variable["dims"],
            attrs=variable.get("attrs"),
        )

    return DatasetSchema(coords=coords, variables=variables)


def dict_to_schema_v3(d):

    coords = {}
    for coord_name, coord in d["coords"].items():
        coords[coord_name] = CoordinateSchema(**coord)

    variables = {}
    for variable_name, variable in d["variables"].items():
        array_spec = variable.pop("array")
        array_spec.update(
            {
                "shape": tuple(array_spec["shape"]),
                "dtype": np.dtype(array_spec["dtype"]),
                "chunks": tuple(array_spec["chunks"]),
            }
        )
        array = ChunkedArray(**array_spec)
        variables[variable_name] = VariableSchema(
            array=array,
            name=variable["name"],
            dims=variable["dims"],
            attrs=variable.get("attrs"),
        )

    return DatasetSchema(coords=coords, variables=variables)


def infer_version(d):
    try:
        return d["version"]
    except KeyError:
        return "v1"


def dict_to_schema(d):
    version = infer_version(d)

    if version == "v1":
        return dict_to_schema_v1_v2(d)
    elif version == "v2alpha":
        return dict_to_schema_v1_v2(d["schema"])
    elif version == "v3":
        return dict_to_schema_v3(d["schema"])
    else:
        raise NotImplementedError(f"Version {version} is not supported.")


def load(fp):
    d = json.load(fp)
    return dict_to_schema(d)


def loads(s):
    fp = io.StringIO(s)
    return load(fp)
