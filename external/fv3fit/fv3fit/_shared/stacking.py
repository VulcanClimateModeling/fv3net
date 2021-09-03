from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import vcm
import xarray as xr
from numpy.random import RandomState
from toolz import curry
from vcm import DerivedMapping, net_heating, net_precipitation, safe
from vcm.catalog import catalog
from vcm.convenience import round_time

from loaders.constants import TIME_NAME

CLOUDS_OFF_TEMP_TENDENCIES = [
    "tendency_of_air_temperature_due_to_longwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_shortwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_turbulence",
    "tendency_of_air_temperature_due_to_dissipation_of_gravity_waves",
]
CLOUDS_OFF_SPHUM_TENDENCIES = ["tendency_of_specific_humidity_due_to_turbulence"]
Z_DIM_NAMES = ["z", "pfull", "z_soil"]
EAST_NORTH_WIND_TENDENCIES = ["dQu", "dQv"]
X_Y_WIND_TENDENCIES = ["dQxwind", "dQywind"]
WIND_ROTATION_COEFFICIENTS = [
    "eastward_wind_u_coeff",
    "eastward_wind_v_coeff",
    "northward_wind_u_coeff",
    "northward_wind_v_coeff",
]
ALLOWED_BROADCAST = ["cos_day", "sin_day", "cos_month", "sin_month"]
SAMPLE_DIM_NAME = "_fv3fit_sample"
DATASET_DIM_NAME = "dataset"
Z_DIM_NAMES = ["z", "pfull"]

Time = str
Tile = int
K = Tuple[Time, Tile]

"""
TODO: Remove the optional sample_dim_name arg from functions once the
stacking sample dim is hard coded and removed as a training function arg.
The presence in the functions below is temporary and done to allow use
of an internal stacking dim (allows inputs to be prestacked in "sample" dim)
"""


class StackedBatches(Sequence[xr.Dataset]):
    def __init__(
        self, batches: Sequence[xr.Dataset], random_state: RandomState,
    ):
        self._batches = batches
        self._random_state = random_state

    def __getitem__(self, idx: Union[int, slice]):
        if isinstance(idx, int):
            return self._stack_batch(self._batches[idx])
        elif isinstance(idx, slice):
            return [self._stack_batch(ds) for ds in self._batches[idx]]
        else:
            raise TypeError(
                f"Invalid argument type of {type(idx)} passed into "
                "StackedBatches.__getitem__."
            )

    def __len__(self) -> int:
        return len(self._batches)

    def _stack_batch(self, ds_unstacked: xr.Dataset) -> xr.Dataset:
        ds = stack_non_vertical(ds_unstacked).load().dropna(dim=SAMPLE_DIM_NAME)
        ds = _check_empty(ds)
        ds = _preserve_samples_per_batch(ds)
        return _shuffled(self._random_state, ds)


def stack_non_vertical(ds: xr.Dataset, sample_dim_name=SAMPLE_DIM_NAME) -> xr.Dataset:
    """
    Stack all dimensions except for the Z dimensions into a sample

    Args:
        ds: dataset with geospatial dimensions
        sample_dim_name: name for new sampling dimension
    """

    ds_group_by_zdim = _group_by_z_dim(ds)
    to_merge = []
    multi_idx = multi_coord_names = None
    for zdim_name, group_ds in ds_group_by_zdim.items():
        stack_dims = [dim for dim in group_ds.dims if dim != zdim_name]
        ds_stacked = safe.stack_once(
            group_ds,
            SAMPLE_DIM_NAME,
            stack_dims,
            allowed_broadcast_dims=[zdim_name] + [TIME_NAME, DATASET_DIM_NAME],
            allowed_broadcast_vars=ALLOWED_BROADCAST,
        )
        if multi_idx is None:
            multi_idx, multi_coord_names = _get_multi_idx(ds_stacked, sample_dim_name)
        # drop multi-level index coordinate for merge
        ds_stacked = ds_stacked.reset_index(sample_dim_name)
        to_merge.append(ds_stacked)

    full_stacked_ds = xr.merge(to_merge)
    # reinsert multi-index
    return (
        full_stacked_ds.reset_coords(multi_coord_names, drop=True)
        .assign_coords({SAMPLE_DIM_NAME: multi_idx})
        .transpose(sample_dim_name, ...)
    )


def _get_multi_idx(ds, stacked_name):
    multi_idx = ds.coords[stacked_name]
    multi_idx_coord_names = [
        name
        for name in multi_idx.reset_index(stacked_name).coords
        if name != stacked_name
    ]
    multi_idx = pd.MultiIndex.from_tuples(multi_idx.values, names=multi_idx_coord_names)

    return multi_idx, multi_idx_coord_names


def _group_by_z_dim(
    ds: xr.Dataset, z_dim_names: Sequence = Z_DIM_NAMES
) -> Mapping[str, xr.Dataset]:
    """
    Cannot stack a dataset with multiple z dimensions. So we'll divide
    and conquer.
    """
    # TODO: Handle case where no_vertical is a single sample variable (e.g. cos day)
    groups = {}
    for varname, da in ds.items():
        da_item = (varname, da)
        da_z_dim = _get_z_dim(da.dims, z_dim_names=z_dim_names)
        if da_z_dim is not None:
            groups.setdefault(da_z_dim, []).append(da_item)
        else:
            groups.setdefault("no_vertical", []).append(da_item)

    for zdim, da_items in groups.items():
        groups[zdim] = xr.Dataset({k: v for k, v in da_items})

    return groups


def _get_z_dim(dims: Sequence, z_dim_names: Sequence = Z_DIM_NAMES) -> Union[str, None]:
    da_z_dim = set(z_dim_names).intersection(dims)
    if len(da_z_dim) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {z_dim_names}.")

    z_dim = da_z_dim.pop() if da_z_dim else None
    return z_dim


def _check_empty(ds: xr.Dataset) -> xr.Dataset:
    """
    Check for an empty variables along a dimension in a dataset
    """
    if len(ds[SAMPLE_DIM_NAME]) == 0:
        raise ValueError("Check for NaN fields in the training data.")
    return ds


def _preserve_samples_per_batch(ds: xr.Dataset) -> xr.Dataset:
    """
    Preserve the approximate number of samples per batch when multiple dataset
    sources are detected in the batch dataset.  Returns an unadjusted dataset
    when no dataset dimension is found.

    Args:
        ds: dataset with sample dimension and potentially a dataset dimension
    """
    try:
        dataset_coord: Optional[xr.DataArray] = ds.coords[DATASET_DIM_NAME]
    except KeyError:
        dataset_coord = None

    if dataset_coord is not None:
        num_datasets = len(set(dataset_coord.values.tolist()))
        ds = ds.thin({SAMPLE_DIM_NAME: num_datasets})

    return ds


@curry
def subsample(
    num_samples: int,
    random_state: np.random.RandomState,
    dataset: xr.Dataset,
    dim=SAMPLE_DIM_NAME,
) -> xr.Dataset:

    """
    Subsample values among a specified dimension

    Args:
        num_samples: number of random sampls to take
        random_state: initialized numpy random state
        dataset: dataset to sample from
        dim (optional): dimension to sample along
    """
    dim_len = dataset.dims[dim]
    sample_idx = random_state.choice(range(dim_len), num_samples, replace=False)
    return dataset.isel({dim: sample_idx})


def _shuffled(random: RandomState, dataset: xr.Dataset) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
        dataset: input data to be shuffled
    """
    chunks_default = (len(dataset[SAMPLE_DIM_NAME]),)
    chunks = dataset.chunks.get(SAMPLE_DIM_NAME, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({SAMPLE_DIM_NAME: shuffled_inds})


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


def _infer_dimension_order(ds: xr.Dataset) -> Tuple:
    # add check here for cases when the dimension order is inconsistent between arrays?
    dim_order = []
    for variable in ds:
        for dim in ds[variable].dims:
            if dim not in dim_order:
                dim_order.append(dim)
    return tuple(dim_order)


def match_prediction_to_input_coords(
    input: xr.Dataset, prediction: xr.Dataset
) -> xr.Dataset:
    # ensure the output coords are the same and dims are same order
    # stack/unstack adds coordinates if none exist before
    input_coords = input.coords
    for key in prediction.coords:
        if key in input_coords:
            prediction.coords[key] = input_coords[key]
        else:
            del prediction.coords[key]
    dim_order = [dim for dim in _infer_dimension_order(input) if dim in prediction.dims]
    return prediction.transpose(*dim_order)


def nonderived_variables(requested: Sequence[Hashable], available: Sequence[Hashable]):
    derived = [var for var in requested if var not in available]
    nonderived = [var for var in requested if var in available]
    # if E/N winds not in underlying data, need to load x/y wind
    # tendencies to derive them
    # TODO move to derived_mapping?
    if any(var in derived for var in EAST_NORTH_WIND_TENDENCIES):
        nonderived += X_Y_WIND_TENDENCIES
    if any(var in derived for var in ["eastward_wind", "northward_wind"]):
        nonderived += ["x_wind", "y_wind"]
    return nonderived


@curry
def add_derived_data(variables: Sequence[str], ds: xr.Dataset) -> xr.Dataset:
    """
    Overlay the DerivedMapping and grab a dataset of specified variables

    Args:
        variables: All variables (derived and non-derived) to include in the
            dataset.
    """
    derived_mapping = DerivedMapping(ds)
    return derived_mapping.dataset(variables)


@curry
def add_grid_info(res: str, ds: xr.Dataset) -> xr.Dataset:
    """
    Add lat, lon, land-type mask information to the dataset

    Args:
        res: grid resolution, format as f'c{number cells in tile}'
    """
    grid = _load_grid(res)
    # Prioritize dataset's land_sea_mask if it differs from grid
    return xr.merge([ds, grid], compat="override")


@curry
def add_wind_rotation_info(res: str, ds: xr.Dataset) -> xr.Dataset:
    """
    Add wind rotation information to the dataset

    Args:
        res: grid resolution, format as f'c{number cells in tile}'
    """

    rotation = _load_wind_rotation_matrix(res).drop("tile")
    common_coords = {"x": ds["x"].values, "y": ds["y"].values}
    rotation = rotation.assign_coords(common_coords)
    return ds.merge(rotation, compat="override")


def _load_grid(res: str) -> xr.Dataset:
    grid = catalog[f"grid/{res}"].to_dask()
    land_sea_mask = catalog[f"landseamask/{res}"].to_dask()
    grid = grid.assign({"land_sea_mask": land_sea_mask["land_sea_mask"]})
    # drop the tiles so that this is compatible with other indexing conventions
    return safe.get_variables(grid, ["lat", "lon", "land_sea_mask"]).drop("tile")


def _load_wind_rotation_matrix(res: str) -> xr.Dataset:
    rotation = catalog[f"wind_rotation/{res}"].to_dask()
    return safe.get_variables(rotation, WIND_ROTATION_COEFFICIENTS)


def get_sample_dataset(mapper):
    sample_key = list(mapper.keys())[0]
    return mapper[sample_key]


def standardize_zarr_time_coord(ds: xr.Dataset) -> xr.Dataset:
    """ Casts a datetime coord to to python datetime and rounds to
    nearest even second (because cftime coords have small rounding
    errors that makes it hard to other datasets join on time)

    Args:
        ds (xr.Dataset): time coordinate is datetime-like object

    Returns:
        xr.Dataset with standardized time coordinates
    """
    # Vectorize doesn't work on type-dispatched function overloading
    times = np.array(list(map(vcm.cast_to_datetime, ds[TIME_NAME].values)))
    times = round_time(times)
    ds = ds.assign_coords({TIME_NAME: times})
    return ds


def preserve_samples_per_batch(
    ds: xr.Dataset, dataset_dim_name=DATASET_DIM_NAME
) -> xr.Dataset:
    """
    Peserve the same-ish number of samples per batch when multiple dataset
    sources are detected in the batch dataset.  Returns an unadjusted dataset
    when no dataset dimension is found.

    Args:
        ds: dataset with sample dimension and potentially a dataset dimension
        dataset_dim_name: name of dataset dimension to check existence of before
            thinning
    """
    try:
        dataset_coord = ds.coords[dataset_dim_name]
    except KeyError:
        dataset_coord = None

    if dataset_coord is not None:
        num_datasets = len(set(dataset_coord.values.tolist()))
        ds = ds.thin({SAMPLE_DIM_NAME: num_datasets})

    return ds


def check_empty(ds: xr.Dataset, dim=SAMPLE_DIM_NAME) -> xr.Dataset:
    """
    Check for an empty variables along a dimension in a dataset
    """
    if len(ds[dim]) == 0:
        raise ValueError("Check for NaN fields in the training data.")
    return ds


@curry
def shuffled(
    random: RandomState, dataset: xr.Dataset, dim=SAMPLE_DIM_NAME
) -> xr.Dataset:
    """
    Shuffles dataset along a dimension within chunks if chunking is present

    Args:
        dim: dimension to shuffle indices along
        random: Initialized random number generator state used for shuffling
        dataset: input data to be shuffled
    """
    chunks_default = (len(dataset[dim]),)
    chunks = dataset.chunks.get(dim, chunks_default)
    chunk_indices = _get_chunk_indices(chunks)
    shuffled_inds = np.concatenate(
        [random.permutation(indices) for indices in chunk_indices]
    )

    return dataset.isel({dim: shuffled_inds})


def net_heating_from_physics(ds: xr.Dataset) -> xr.DataArray:

    fluxes = (
        ds["total_sky_downward_longwave_flux_at_surface"],
        ds["total_sky_downward_shortwave_flux_at_surface"],
        ds["total_sky_upward_longwave_flux_at_surface"],
        ds["total_sky_upward_longwave_flux_at_top_of_atmosphere"],
        ds["total_sky_upward_shortwave_flux_at_surface"],
        ds["total_sky_upward_shortwave_flux_at_top_of_atmosphere"],
        ds["total_sky_downward_shortwave_flux_at_top_of_atmosphere"],
        ds["sensible_heat_flux"],
        ds["surface_precipitation_rate"],
    )
    return net_heating(*fluxes)


def net_precipitation_from_physics(ds: xr.Dataset) -> xr.DataArray:

    fluxes = (
        ds["latent_heat_flux"],
        ds["surface_precipitation_rate"],
    )
    return net_precipitation(*fluxes)


def assign_net_physics_terms(ds: xr.Dataset) -> xr.Dataset:
    net_terms: Mapping[Hashable, Any] = {
        "net_heating": net_heating_from_physics(ds),
        "net_precipitation": net_precipitation_from_physics(ds),
    }
    return ds.assign(net_terms)
