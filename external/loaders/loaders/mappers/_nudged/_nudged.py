import logging
import xarray as xr
import os
from typing import Hashable, Sequence, Mapping, Optional, Any, MutableMapping
import fsspec
import zarr

from .._base import MultiDatasetMapper
from .._xarray import XarrayMapper
from loaders._config import mapper_functions
from loaders.typing import Mapper

logger = logging.getLogger(__name__)

Z_DIM_NAME = "z"

Time = str
Dataset = MutableMapping[Hashable, Any]


@mapper_functions.register
def open_nudge_to_obs(
    data_path: str,
    nudging_tendency_variables: Optional[Mapping[str, str]] = None,
    physics_timestep_seconds: float = 900.0,
    consolidated: bool = True,
) -> Mapper:
    """
    Load nudge-to-obs data mapper for use with training. Merges
    variables saved in the physics tendencies, nudging tendencies (Fortran
    diagnostics), and model state zarrs.

    Because the nudge-to-obs routine conducts nudging within the physics step,
    the returned physics tendency is computed as the output physics_tendency minus
    the nudging tendency. Similarly, because model states are output at the end
    of the timestep, the nudging increment is subtracted to return the
    ``before nudging`` state for training.

    Args:
        data_path (str): path to a nudge-to-obs output directory, remote or local
        nudging_tendency_variables: (optional): mapping of variables to their renamed
            nudging tendencies. Defaults to
            {"air_temperature": "dQ1", "specific_humidity": "dQ2"}
        physics_timestep_seconds (float): physics timestep, i.e., dt_atmos; defaults
            to 900.0
        consolidated (bool): whether zarrs to open have consolidated metadata

    Returns:
        mapper to dataset containing nudging tendencies, physics tendencies,
            and model state data

    """

    datasets = _get_datasets(
        data_path,
        [
            "physics_tendencies.zarr",
            "nudging_tendencies.zarr",
            "state_after_timestep.zarr",
        ],
        consolidated=consolidated,
    )

    ds = xr.merge(
        [
            datasets["physics_tendencies.zarr"].rename(
                {
                    "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
                    "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
                    "tendency_of_eastward_wind_due_to_fv3_physics": "pQu",
                    "tendency_of_northward_wind_due_to_fv3_physics": "pQv",
                }
            ),
            datasets["nudging_tendencies.zarr"].rename(
                {
                    "t_dt_nudge": "dQ1",
                    "q_dt_nudge": "dQ2",
                    "u_dt_nudge": "dQu",
                    "v_dt_nudge": "dQv",
                    "grid_xt": "x",
                    "grid_yt": "y",
                    "pfull": "z",
                }
            ),
            datasets["state_after_timestep.zarr"],
        ]
    )

    nudging_tendency_variables = nudging_tendency_variables or {
        "air_temperature": "dQ1",
        "specific_humidity": "dQ2",
        "eastward_wind": "dQu",
        "northward_wind": "dQv",
    }

    differenced_state: Dataset = {}
    for (
        nudging_variable_name,
        nudging_tendency_name,
    ) in nudging_tendency_variables.items():
        differenced_state[nudging_variable_name] = (
            ds[nudging_variable_name]
            - ds[nudging_tendency_name] * physics_timestep_seconds
        )
    ds = ds.assign(differenced_state)

    differenced_physics_tendency: Dataset = {}
    for nudging_name, physics_name in zip(
        ["dQ1", "dQ2", "dQu", "dQv"], ["pQ1", "pQ2", "pQu", "pQv"]
    ):
        differenced_physics_tendency[physics_name] = ds[physics_name] - ds[nudging_name]
    ds = ds.assign(differenced_physics_tendency)

    return XarrayMapper(ds)


@mapper_functions.register
def open_nudge_to_fine(
    data_path: str,
    nudging_variables: Sequence[str],
    physics_timestep_seconds: float = 900.0,
    consolidated: bool = True,
    datasets: Sequence[str] = (
        "physics_tendencies.zarr",
        "nudging_tendencies.zarr",
        "state_after_timestep.zarr",
    ),
    cache_size_mb: Optional[float] = None,
) -> XarrayMapper:
    """
    Load nudge-to-fine data mapper for use with training. Merges
    variables saved in the physics tendencies, nudging tendencies, and
    model state zarrs.
    Because model states are output at the end of the timestep, the nudging
    increment is subtracted to return the ``before nudging`` state for training.
    Args:
        url (str):  path to nudge-to-fine output directory, remote or local
        nudging_variables (Sequence[str]): Names of nudged variables, nudging tendency
            will be subtracted to retrieve model state before nudging
        physics_timestep_seconds (float): physics timestep, i.e., dt_atmos; defaults
            to 900.0
        consolidated (bool): whether zarrs to open have consolidated metadata
        datasets: names of zarrs at the given URL to include, defaults are
            physics_tendencies.zarr, nudging_tendencies.zarr, and
            state_after_timestep.zarr (which you should probably include).
            For example, you may want to include also "diags.zarr" to retrieve
            total_precipitation_rate.
         cache_size_mb: Cache size in MB for using zarr.storage.LRUStoreCache
            for accessing data. A cache of this size is created for each zarr
            dataset in the datasets arg. No LRU caches created if this arg is not
            supplied.
    Returns:
        mapper to dataset containing nudging tendencies, physics tendencies,
            and model state data
    """

    ds = xr.merge(
        _get_datasets(
            data_path, datasets, consolidated=consolidated, cache_size_mb=cache_size_mb
        ).values(),
        join="inner",
    )

    differenced_state: Dataset = {}
    for nudging_variable in nudging_variables:
        nudging_tendency = ds[f"{nudging_variable}_tendency_due_to_nudging"]
        differenced_state[nudging_variable] = (
            ds[nudging_variable] - nudging_tendency * physics_timestep_seconds
        )
    ds = ds.assign(differenced_state)

    rename_vars: Mapping[Hashable, Hashable] = {
        "air_temperature_tendency_due_to_nudging": "dQ1",
        "specific_humidity_tendency_due_to_nudging": "dQ2",
        "x_wind_tendency_due_to_nudging": "dQxwind",
        "y_wind_tendency_due_to_nudging": "dQywind",
        "eastward_wind_tendency_due_to_nudging": "dQu",
        "northward_wind_tendency_due_to_nudging": "dQv",
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
        "tendency_of_eastward_wind_due_to_fv3_physics": "pQu",
        "tendency_of_northward_wind_due_to_fv3_physics": "pQv",
    }
    rename_vars = {k: v for k, v in rename_vars.items() if k in ds}
    return XarrayMapper(ds.rename(rename_vars))


@mapper_functions.register
def open_nudge_to_fine_multiple_datasets(
    data_path: str,
    additional_paths: Sequence[str],
    names: Optional[Sequence[Hashable]] = None,
    **kwargs,
) -> Mapper:
    """
    Load sequence of mappers to nudged datasets containing dQ tendency terms.

    Args:
        data_path: path to directory with nudging output
        additional_paths: additional paths to directories with nudging output
        names: sequence of dataset names, starting with data_path and
            followed by additional_paths in order.
            gets assigned as the "dataset" coordinate
        **kwargs: keyword arguments passed to open_nudge_to_fine

    Returns
        merged_nudged: mapper of timestamps to Dataset containing tendency terms
            with a "dataset" dimension
    """
    paths = [data_path]
    paths.extend(additional_paths)
    mappers = [open_nudge_to_fine(path, **kwargs) for path in paths]
    return MultiDatasetMapper(mappers, names=names)


def _get_datasets(
    url: str,
    sources: Sequence[str],
    consolidated: bool = True,
    cache_size_mb: Optional[float] = None,
) -> MutableMapping[Hashable, xr.Dataset]:
    datasets: MutableMapping[Hashable, xr.Dataset] = {}
    for source in sources:
        mapper = fsspec.get_mapper(os.path.join(url, f"{source}"))
        if cache_size_mb is not None:
            mapper = zarr.LRUStoreCache(mapper, max_size=int(cache_size_mb * 1e6),)
        ds = xr.open_zarr(mapper, consolidated=consolidated)
        datasets[source] = ds
    return datasets


@mapper_functions.register
def open_nudge_to_fine_scream(
    data_path: str,
    nudging_variables: Sequence[str],
    physics_timestep_seconds: float = 900.0,
    consolidated: bool = True,
    datasets: Sequence[str] = (
        "physics_tendencies.zarr",
        "nudging_tendencies.zarr",
        "state_after_timestep.zarr",
    ),
    cache_size_mb: Optional[float] = None,
) -> XarrayMapper:
    """
    Similar to open_nudge_to_fine, but for scream specifically.

    Args:
        url (str):  path to nudge-to-fine output directory, remote or local
        nudging_variables (Sequence[str]): Names of nudged variables, nudging tendency
            will be subtracted to retrieve model state before nudging
        physics_timestep_seconds (float): physics timestep, i.e., dt_atmos; defaults
            to 900.0
        consolidated (bool): whether zarrs to open have consolidated metadata
        datasets: names of zarrs at the given URL to include, defaults are
            physics_tendencies.zarr, nudging_tendencies.zarr, and
            state_after_timestep.zarr (which you should probably include).
            For example, you may want to include also "diags.zarr" to retrieve
            total_precipitation_rate.
         cache_size_mb: Cache size in MB for using zarr.storage.LRUStoreCache
            for accessing data. A cache of this size is created for each zarr
            dataset in the datasets arg. No LRU caches created if this arg is not
            supplied.

    Returns:
        mapper to dataset containing nudging tendencies, physics tendencies,
            and model state data
    """

    ds = xr.merge(
        _get_datasets(
            data_path, datasets, consolidated=consolidated, cache_size_mb=cache_size_mb
        ).values(),
        join="inner",
    )

    differenced_state: Dataset = {}
    for nudging_variable in nudging_variables:
        nudging_tendency = ds[f"{nudging_variable}_tendency_due_to_nudging"]
        differenced_state[nudging_variable] = (
            ds[nudging_variable] - nudging_tendency * physics_timestep_seconds
        )
    ds = ds.assign(differenced_state)

    rename_vars: Mapping[Hashable, Hashable] = {
        "T_mid_tendency_due_to_nudging": "dQ1",
        "qv_tendency_due_to_nudging": "dQ2",
        "U_tendency_due_to_nudging": "dQu",
        "V_tendency_due_to_nudging": "dQv",
        "tendency_of_U_due_to_scream_physics": "pQu",
        "tendency_of_V_due_to_scream_physics": "pQv",
        "tendency_of_T_mid_due_to_scream_physics": "pQ1",
        "tendency_of_qv_due_to_scream_physics": "pQ2",
        "LW_flux_dn_at_model_bot": "total_sky_downward_longwave_flux_at_surface",
        "LW_flux_up_at_model_bot": "total_sky_upward_longwave_flux_at_surface",
        "LW_flux_up_at_model_top": "total_sky_upward_longwave_flux_at_top_of_atmosphere",
        "SW_flux_dn_at_model_bot": "total_sky_downward_shortwave_flux_at_surface",
        "SW_flux_up_at_model_bot": "total_sky_upward_shortwave_flux_at_surface",
        "SW_flux_up_at_model_top": "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
        "SW_flux_dn_at_model_top": "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    }
    rename_vars = {k: v for k, v in rename_vars.items() if k in ds}
    return XarrayMapper(ds.rename(rename_vars))
