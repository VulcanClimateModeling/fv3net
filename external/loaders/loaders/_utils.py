import numpy as np
from numpy.random import RandomState
from typing import Any, Hashable, Mapping, Optional, Tuple, Sequence
from toolz.functoolz import curry
import xarray as xr
import vcm
from vcm import safe, net_heating, net_precipitation, DerivedMapping
from vcm.convenience import round_time

from .constants import DATASET_DIM_NAME, SAMPLE_DIM_NAME, TIME_NAME
from vcm.catalog import catalog


CLOUDS_OFF_TEMP_TENDENCIES = [
    "tendency_of_air_temperature_due_to_longwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_shortwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_turbulence",
    "tendency_of_air_temperature_due_to_dissipation_of_gravity_waves",
]
CLOUDS_OFF_SPHUM_TENDENCIES = ["tendency_of_specific_humidity_due_to_turbulence"]
Z_DIM_NAMES = ["z", "pfull"]
EAST_NORTH_WIND_TENDENCIES = ["dQu", "dQv"]
X_Y_WIND_TENDENCIES = ["dQxwind", "dQywind"]
WIND_ROTATION_COEFFICIENTS = [
    "eastward_wind_u_coeff",
    "eastward_wind_v_coeff",
    "northward_wind_u_coeff",
    "northward_wind_v_coeff",
]

Time = str
Tile = int
K = Tuple[Time, Tile]


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

    rotation = _load_wind_rotation_matrix(res)
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


def stack_non_vertical(ds: xr.Dataset, sample_dim_name=SAMPLE_DIM_NAME) -> xr.Dataset:
    """
    Stack all dimensions except for the Z dimensions into a sample

    Args:
        ds: dataset with geospatial dimensions
        sample_dim_name: name for new sampling dimension
    """
    stack_dims = [dim for dim in ds.dims if dim not in Z_DIM_NAMES]
    if len(set(ds.dims).intersection(Z_DIM_NAMES)) > 1:
        raise ValueError("Data cannot have >1 feature dimension in {Z_DIM_NAMES}.")
    ds_stacked = safe.stack_once(
        ds,
        sample_dim_name,
        stack_dims,
        allowed_broadcast_dims=Z_DIM_NAMES + [TIME_NAME, DATASET_DIM_NAME],
    )
    return ds_stacked.transpose()


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
        dataset_coord: Optional[xr.DataArray] = ds.coords[dataset_dim_name]
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


def _get_chunk_indices(chunks):
    indices = []

    start = 0
    for chunk in chunks:
        indices.append(list(range(start, start + chunk)))
        start += chunk
    return indices


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
