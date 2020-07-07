from datetime import datetime
import functools
import logging
from numpy.random import RandomState
import pandas as pd
from typing import Iterable, Sequence, Mapping, Any, Hashable, Optional
import xarray as xr
from vcm import safe
from toolz import partition, compose
from ._sequences import FunctionOutputSequence
from .._utils import stack_dropnan_shuffle, load_grid, add_cosine_zenith_angle
from ..constants import TIME_NAME, TIME_FMT, COS_Z_VAR
import loaders

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def batches_from_geodata(
    data_path: str,
    variable_names: Iterable[str],
    mapping_function: str,
    mapping_kwargs: Optional[Mapping[str, Any]] = None,
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    init_time_dim_name: str = "time",
    rename_variables: Optional[Mapping[str, str]] = None,
    timesteps: Optional[Sequence[str]] = None,
) -> Sequence[xr.Dataset]:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train. The data is assumed to
    have geospatial dimensions and is accessed through a mapper interface.


    Args:
        data_path (str): Path to data store to be loaded via mapper.
        variable_names (Iterable[str]): data variables to select
        mapping_function (str): Name of a callable which opens a mapper to the data
        mapping_kwargs (Mapping[str, Any]): mapping of keyword arguments to be
            passed to the mapping function
        timesteps_per_batch (int, optional): Defaults to 1.
        random_seed (int, optional): Defaults to 0.
        init_time_dim_name (str, optional): Name of time dim in data source.
            Defaults to "time".
        rename_variables (Mapping[str, str], optional): Defaults to None.
        
    Raises:
        TypeError: If no variable_names are provided to select the final datasets
        
    Returns:
        Sequence of xarray datasets for use in training batches.
    """
    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)
    batches = batches_from_mapper(
        data_mapping,
        variable_names,
        timesteps_per_batch,
        random_seed,
        init_time_dim_name,
        rename_variables,
        timesteps,
    )
    return batches


def _create_mapper(
    data_path, mapping_func_name: str, mapping_kwargs: Mapping[str, Any]
) -> Mapping[str, xr.Dataset]:
    mapping_func = getattr(loaders.mappers, mapping_func_name)
    mapping_kwargs = mapping_kwargs or {}
    return mapping_func(data_path, **mapping_kwargs)


def batches_from_mapper(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Iterable[str],
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    init_time_dim_name: str = "time",
    rename_variables: Optional[Mapping[str, str]] = None,
    timesteps: Optional[Sequence[str]] = None,
) -> Sequence[xr.Dataset]:
    """ The function returns a sequence of datasets that is later
    iterated over in  ..sklearn.train.
    
    Args:
        data_mapping (Mapping[str, xr.Dataset]): Interface to select data for
            given timestep keys.
        variable_names (Iterable[str]): data variables to select
        timesteps_per_batch (int, optional): Defaults to 1.
        random_seed (int, optional): Defaults to 0.
        init_time_dim_name (str, optional): Name of time dim in data source.
            Defaults to "time".
        rename_variables (Mapping[str, str], optional): Defaults to None.
        timesteps: List of timesteps to use in training.
    Raises:
        TypeError: If no variable_names are provided to select the final datasets
        
    Returns:
        Sequence of xarray datasets
    """
    if timesteps and set(timesteps).issubset(data_mapping.keys()) is False:
        raise ValueError(
            "Timesteps specified in file are not present in data: "
            f"{list(set(timesteps)-set(data_mapping.keys()))}"
        )

    random_state = RandomState(random_seed)
    if rename_variables is None:
        rename_variables = {}
    if len(variable_names) == 0:
        raise TypeError("At least one value must be given for variable_names")

    timesteps = timesteps or data_mapping.keys()
    num_times = len(timesteps)
    times = _sample(timesteps, num_times, random_state)
    batched_timesteps = list(partition(timesteps_per_batch, times))

    load_batch = functools.partial(
        _load_batch, data_mapping, variable_names, rename_variables, init_time_dim_name,
    )

    transform = functools.partial(
        stack_dropnan_shuffle, init_time_dim_name, random_state
    )
    if COS_Z_VAR in variable_names:
        grid = load_grid()
        insert_cos_z = functools.partial(add_cosine_zenith_angle, grid)
        batch_func = compose(transform, insert_cos_z, load_batch)
    else:
        batch_func = compose(transform, load_batch)

    seq = FunctionOutputSequence(batch_func, batched_timesteps)
    seq.attrs["times"] = times

    return seq


def diagnostic_batches_from_geodata(
    data_path: str,
    variable_names: Sequence[str],
    mapping_function: str,
    mapping_kwargs: Optional[Mapping[str, Any]] = None,
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    init_time_dim_name: str = "time",
    rename_variables: Optional[Mapping[str, str]] = None,
    timesteps: Optional[Sequence[str]] = None,
) -> Sequence[xr.Dataset]:
    """Load a dataset sequence for dagnostic purposes. Uses the same batch subsetting as
    as batches_from_mapper but without transformation and stacking
    
    Args:
        data_path: Path to data store to be loaded via mapper.
        variable_names (Iterable[str]): data variables to select
        mapping_function (str): Name of a callable which opens a mapper to the data
        mapping_kwargs (Mapping[str, Any]): mapping of keyword arguments to be
            passed to the mapping function
        timesteps_per_batch (int, optional): Defaults to 1.
        num_batches (int, optional): Defaults to None.
        random_seed (int, optional): Defaults to 0.
        init_time_dim_name (str, optional): Name of time dim in data source.
            Defaults to "time".
        rename_variables (Mapping[str, str], optional): Defaults to None.
        timesteps: List of timesteps to use in training.

    Raises:
        TypeError: If no variable_names are provided to select the final datasets
        
    Returns:
        Sequence of xarray datasets for use in training batches.
    """

    data_mapping = _create_mapper(data_path, mapping_function, mapping_kwargs)
    sequence = diagnostic_batches_from_mapper(
        data_mapping,
        variable_names,
        timesteps_per_batch,
        random_seed,
        init_time_dim_name,
        rename_variables,
        timesteps,
    )

    return sequence


def diagnostic_batches_from_mapper(
    data_mapping: Mapping[str, xr.Dataset],
    variable_names: Sequence[str],
    timesteps_per_batch: int = 1,
    random_seed: int = 0,
    init_time_dim_name: str = "time",
    rename_variables: Mapping[str, str] = None,
    timesteps: Sequence[str] = None,
) -> Sequence[xr.Dataset]:
    if timesteps and set(timesteps).issubset(data_mapping.keys()) is False:
        raise ValueError(
            "Timesteps specified in file are not present in data: "
            f"{list(set(timesteps)-set(data_mapping.keys()))}"
        )
    random_state = RandomState(random_seed)
    if rename_variables is None:
        rename_variables = {}
    timesteps = timesteps or data_mapping.keys()
    num_times = len(timesteps)
    times = _sample(timesteps, num_times, random_state)
    batched_timesteps = list(partition(timesteps_per_batch, times))

    load_batch = functools.partial(
        _load_batch, data_mapping, variable_names, rename_variables, init_time_dim_name,
    )
    if COS_Z_VAR in variable_names:
        grid = load_grid()
        insert_cos_z = functools.partial(add_cosine_zenith_angle, grid)
        batch_func = compose(insert_cos_z, load_batch)
    else:
        batch_func = load_batch
    seq = FunctionOutputSequence(batch_func, batched_timesteps)
    seq.attrs["times"] = times
    return seq


def _sample(seq: Sequence[Any], n: int, random_state: RandomState) -> Sequence[Any]:
    return random_state.choice(list(seq), n, replace=False).tolist()


def _load_batch(
    mapper: Mapping[str, xr.Dataset],
    data_vars: Iterable[str],
    rename_variables: Mapping[str, str],
    init_time_dim_name: str,
    keys: Iterable[Hashable],
) -> xr.Dataset:
    time_coords = [datetime.strptime(key, TIME_FMT) for key in keys]
    ds = xr.concat(
        [mapper[key] for key in keys], pd.Index(time_coords, name=init_time_dim_name)
    )
    ds = ds.rename(rename_variables)

    # cos z is special case of feature that is not present in dataset
    ds = safe.get_variables(ds, [var for var in data_vars if var != COS_Z_VAR])
    return ds
