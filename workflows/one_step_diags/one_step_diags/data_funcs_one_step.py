import xarray as xr
import numpy as np
from scipy.stats import binned_statistic
from vcm.select import mask_to_surface_type
from vcm.convenience import parse_datetime_from_str

# from vcm.safe import get_variables
from vcm import thermo, local_time, net_precipitation
import logging
from typing import Sequence, Mapping
from .config import SFC_VARIABLES
from .constants import (
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    DELTA_DIM,
    ZARR_STEP_DIM,
    ZARR_STEP_NAMES,
)

from fv3net.pipelines.common import load_hires_prog_diag
from fv3net.diagnostics.data import net_heating_from_dataset


logger = logging.getLogger("one_step_diags")

_KG_M2S_TO_MM_DAY = (1e3 * 86400) / 997.0


def time_coord_to_datetime(
    ds: xr.Dataset, time_coord: str = INIT_TIME_DIM
) -> xr.Dataset:

    init_datetime_coords = [
        parse_datetime_from_str(timestamp) for timestamp in ds[time_coord].values
    ]
    ds = ds.assign_coords({time_coord: init_datetime_coords})

    return ds


def insert_hi_res_diags(
    ds: xr.Dataset, hi_res_diags_path: str, varnames_mapping: Mapping
) -> xr.Dataset:

    # temporary kluge for cumulative surface longwave from coarse diag netcdfs
    cumulative_vars = ["DLWRFsfc", "ULWRFsfc"]
    surface_longwave = (
        ds[cumulative_vars].fillna(value=0).diff(dim=FORECAST_TIME_DIM, label="upper")
    )
    ds = ds.drop(labels=cumulative_vars).merge(surface_longwave)

    new_dims = {"grid_xt": "x", "grid_yt": "y", "initialization_time": INIT_TIME_DIM}
    datetimes = list(ds[INIT_TIME_DIM].values)
    ds_hires_diags = (
        load_hires_prog_diag(hi_res_diags_path, datetimes).rename(new_dims).load()
    )

    new_vars = {}
    for coarse_name, hires_name in varnames_mapping.items():
        hires_name = hires_name + "_coarse"  # this is confusing...
        hires_var = ds_hires_diags[hires_name].transpose(
            INIT_TIME_DIM, "tile", "y", "x"
        )
        coarse_var = ds[coarse_name].load()
        coarse_var.loc[
            {FORECAST_TIME_DIM: 0, ZARR_STEP_DIM: ZARR_STEP_NAMES["begin"]}
        ] = hires_var
        new_vars[coarse_name] = coarse_var

    ds = ds.assign(new_vars)

    return ds


def insert_derived_vars_from_ds_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Add derived vars (combinations of direct output variables) to dataset"""

    cloud_water_ice_mixing_ratio = (
        ds["cloud_ice_mixing_ratio"] + ds["cloud_water_mixing_ratio"]
    )
    cloud_water_ice_mixing_ratio.attrs.update(
        {"long_name": "cloud water and ice mixing ratio", "units": "kg/kg"}
    )

    precipitating_water_mixing_ratio = (
        ds["rain_mixing_ratio"] + ds["snow_mixing_ratio"] + ds["graupel_mixing_ratio"]
    )
    precipitating_water_mixing_ratio.attrs.update(
        {"long_name": "precipitating water mixing ratio", "units": "kg/kg"}
    )

    ds = ds.assign(
        {
            "total_water": thermo.total_water(
                ds["specific_humidity"],
                ds["cloud_ice_mixing_ratio"],
                ds["cloud_water_mixing_ratio"],
                ds["rain_mixing_ratio"],
                ds["snow_mixing_ratio"],
                ds["graupel_mixing_ratio"],
            ),
            "precipitating_water": precipitating_water_mixing_ratio,
            "cloud_water_ice": cloud_water_ice_mixing_ratio,
            "liquid_ice_temperature": thermo.liquid_ice_temperature(
                ds["air_temperature"],
                ds["cloud_ice_mixing_ratio"],
                ds["cloud_water_mixing_ratio"],
                ds["rain_mixing_ratio"],
                ds["snow_mixing_ratio"],
                ds["graupel_mixing_ratio"],
            ),
            "surface_pressure": thermo.surface_pressure_from_delp(
                ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "precipitable_water": thermo.precipitable_water(
                ds["specific_humidity"], ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "column_integrated_heat": thermo.column_integrated_heat(
                ds["air_temperature"], ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "net_precipitation_physics": net_precipitation(
                ds["latent_heat_flux"], ds["total_precipitation"]
            ),
            "evaporation": thermo.surface_evaporation_mm_day_from_latent_heat_flux(
                ds["latent_heat_flux"]
            ),
            "net_heating_physics": net_heating_from_dataset(
                ds.rename(
                    {
                        "sensible_heat_flux": "SHTFLsfc",
                        "total_precipitation": "PRATEsfc",
                    }
                )
            ),
            "total_precipitation": (
                (ds["total_precipitation"] * _KG_M2S_TO_MM_DAY).assign_attrs(
                    {"long name": "total precipitation", "units": "mm/day"}
                )
            ),
        }
    )

    return ds.drop(
        labels=[
            "cloud_ice_mixing_ratio",
            "cloud_water_mixing_ratio",
            "rain_mixing_ratio",
            "snow_mixing_ratio",
            "graupel_mixing_ratio",
        ]
        + SFC_VARIABLES
    )


def _align_time_and_concat(ds_hires: xr.Dataset, ds_coarse: xr.Dataset) -> xr.Dataset:

    ds_coarse = (
        ds_coarse.isel({INIT_TIME_DIM: [0]})
        .assign_coords({DELTA_DIM: "coarse"})
        .drop(labels=ZARR_STEP_DIM)
        .squeeze()
    )

    ds_hires = (
        ds_hires.isel({FORECAST_TIME_DIM: [0]})
        .sel({ZARR_STEP_DIM: ZARR_STEP_NAMES["begin"]})
        .drop(labels=ZARR_STEP_DIM)
        .assign_coords({DELTA_DIM: "hi-res"})
        .squeeze()
    )

    dim_excl = [dim for dim in ds_hires.dims if dim != FORECAST_TIME_DIM]
    ds_hires = ds_hires.broadcast_like(ds_coarse, exclude=dim_excl)

    return xr.concat([ds_hires, ds_coarse], dim=DELTA_DIM)


def _compute_both_tendencies_and_concat(ds: xr.Dataset) -> xr.Dataset:

    dt_init = ds[INIT_TIME_DIM].diff(INIT_TIME_DIM).isel(
        {INIT_TIME_DIM: 0}
    ) / np.timedelta64(1, "s")
    tendencies_hires = ds.diff(INIT_TIME_DIM, label="lower") / dt_init

    dt_forecast = (
        ds[FORECAST_TIME_DIM].diff(FORECAST_TIME_DIM).isel({FORECAST_TIME_DIM: 0})
    )
    tendencies_coarse = ds.diff(ZARR_STEP_DIM, label="lower") / dt_forecast

    tendencies_both = _align_time_and_concat(tendencies_hires, tendencies_coarse)

    return tendencies_both


def _select_both_states_and_concat(ds: xr.Dataset) -> xr.Dataset:

    states_hires = ds.isel({INIT_TIME_DIM: slice(None, -1)})
    states_coarse = ds.sel({ZARR_STEP_DIM: ZARR_STEP_NAMES["begin"]})
    states_both = _align_time_and_concat(states_hires, states_coarse)

    return states_both


def get_states_and_tendencies(ds: xr.Dataset) -> xr.Dataset:

    tendencies = _compute_both_tendencies_and_concat(ds)
    states = _select_both_states_and_concat(ds)
    states_and_tendencies = xr.concat([states, tendencies], dim="var_type")
    states_and_tendencies = states_and_tendencies.assign_coords(
        {"var_type": ["states", "tendencies"]}
    )

    return states_and_tendencies


def insert_column_integrated_tendencies(ds: xr.Dataset) -> xr.Dataset:

    ds = ds.assign(
        {
            "column_integrated_heating": thermo.column_integrated_heating(
                ds["air_temperature"].sel({"var_type": "tendencies"}),
                ds["pressure_thickness_of_atmospheric_layer"].sel(
                    {"var_type": "states"}
                ),
            ).expand_dims({"var_type": ["tendencies"]}),
            "minus_column_integrated_moistening": (
                thermo.minus_column_integrated_moistening(
                    ds["specific_humidity"].sel({"var_type": "tendencies"}),
                    ds["pressure_thickness_of_atmospheric_layer"].sel(
                        {"var_type": "states"}
                    ),
                ).expand_dims({"var_type": ["tendencies"]})
            ),
        }
    )

    return ds


def insert_model_run_differences(ds: xr.Dataset) -> xr.Dataset:

    ds_residual = (
        ds.sel({DELTA_DIM: "hi-res"}) - ds.sel({DELTA_DIM: "coarse"})
    ).assign_coords({DELTA_DIM: "hi-res - coarse"})

    # combine into one dataset
    return xr.concat([ds, ds_residual], dim=DELTA_DIM)


def insert_abs_vars(ds: xr.Dataset, varnames: Sequence) -> xr.Dataset:

    for var in varnames:
        if var in ds:
            ds[var + "_abs"] = np.abs(ds[var])
            ds[var + "_abs"].attrs.update(
                {
                    "long_name": f"absolute {ds[var].attrs['long_name']}"
                    if "long_name" in ds[var].attrs
                    else None,
                    "units": ds[var].attrs["units"],
                }
            )
        else:
            raise ValueError("Invalid variable name for absolute tendencies.")

    return ds


def insert_variable_at_model_level(
    ds: xr.Dataset, varnames: Sequence, levels: Sequence
):

    for var in varnames:
        if var in ds:
            for level in levels:
                new_name = f"{var}_level_{level}"
                ds = ds.assign({new_name: ds[var].sel({"z": level})})
                ds[new_name].attrs.update(
                    {"long_name": f"{var} at model level {level}"}
                )
        else:
            raise ValueError("Invalid variable for model level selection.")

    return ds


def mean_diurnal_cycle(
    da: xr.DataArray, local_time: xr.DataArray, stack_dims: list = ["x", "y", "tile"]
) -> xr.DataArray:

    local_time = (
        local_time.stack(dimensions={"sample": stack_dims}).load().dropna("sample")
    )
    da = da.stack(dimensions={"sample": stack_dims}).load().dropna("sample")
    other_dims = [dim for dim in da.dims if dim != "sample"]
    diurnal_coords = [(dim, da[dim]) for dim in other_dims] + [
        ("local_hour_of_day", np.arange(0.0, 24.0))
    ]
    diurnal_da = xr.DataArray(coords=diurnal_coords)
    diurnal_da.name = da.name
    for valid_time in da[FORECAST_TIME_DIM]:
        da_single_time = da.sel({FORECAST_TIME_DIM: valid_time})
        try:
            bin_means, bin_edges, _ = binned_statistic(
                local_time.values, da_single_time.values, bins=np.arange(0.0, 25.0)
            )
            diurnal_da.loc[{FORECAST_TIME_DIM: valid_time}] = bin_means
        except AttributeError:
            logger.warn(
                f"Diurnal mean computation failed for initial time "
                f"{da[INIT_TIME_DIM].item()} for {da.name} "
                f"and forecast time {valid_time.item()} due to null values."
            )
            diurnal_da = None
            break

    return diurnal_da


def insert_diurnal_means(
    ds: xr.Dataset, var_mapping: Mapping, mask: str = "land_sea_mask",
) -> xr.Dataset:

    ds = ds.assign({"local_time": local_time(ds, time=INIT_TIME_DIM)})

    for domain in ["global", "land", "sea"]:

        logger.info(f"Computing diurnal means for {domain}")

        if domain in ["land", "sea"]:
            ds_domain = mask_to_surface_type(ds, domain, surface_type_var=mask)
        else:
            ds_domain = ds

        for var, attrs in var_mapping.items():

            residual_name = attrs["hi-res - coarse"]["name"]
            residual_type = attrs["hi-res - coarse"]["var_type"]
            physics_name = attrs["physics"]["name"]
            physics_type = attrs["physics"]["var_type"]

            da_residual_domain = mean_diurnal_cycle(
                ds_domain[residual_name].sel(
                    {DELTA_DIM: ["hi-res - coarse"], "var_type": residual_type}
                ),
                ds_domain["local_time"],
            )
            da_physics_domain = mean_diurnal_cycle(
                ds_domain[physics_name].sel(
                    {DELTA_DIM: ["hi-res", "coarse"], "var_type": physics_type}
                ),
                ds_domain["local_time"],
            )
            if da_residual_domain is None or da_physics_domain is None:
                return None
            else:
                ds = ds.assign(
                    {
                        f"{var}_{domain}": xr.concat(
                            [da_physics_domain, da_residual_domain], dim=DELTA_DIM
                        )
                    }
                )
                ds[f"{var}_{domain}"].attrs.update(ds[residual_name].attrs)

    return ds


def _mask_to_PminusE_sign(
    ds: xr.Dataset, sign: str, PminusE_varname: str, PminusE_dataset: str = "hi-res"
) -> xr.Dataset:
    """
    Args:
        ds: xarray dataset
        sign: one of ['positive', 'negative']
        PminusE_varname: Name of the P - E var in ds.
        PminusE_dataset: Name of the P - E dataset to use, optional.
            Defaults to 'hi-res'.
    Returns:
        input dataset masked to the P - E sign specified
    """
    if sign in ["none", "None", None]:
        logger.info("surface_type provided as None: no mask applied.")
        return ds
    elif sign not in ["positive", "negative"]:
        raise ValueError("Must mask to either positive or negative.")

    PminusE = ds[PminusE_varname].sel({DELTA_DIM: "hi-res"})
    mask = PminusE > 0 if sign == "positive" else PminusE < 0

    return ds.where(mask)


def _weighted_mean(
    ds: xr.Dataset, weights: xr.DataArray, dims=["tile", "x", "y"]
) -> xr.Dataset:
    """Compute weighted mean of a dataset
    Args:
        ds: dataset to be averaged
        weights: xr.DataArray of weights of the dataset, must be broadcastable to ds
        dims: dimension names over which to average
    Returns
        weighted mean average ds
    """

    return (ds * weights).sum(dim=dims) / weights.sum(dim=dims)


def insert_area_means(
    ds: xr.Dataset, weights: xr.DataArray, varnames: list, mask_names: list
) -> xr.Dataset:

    wm = _weighted_mean(ds, weights)
    for var in varnames:
        if var in ds:
            new_name = f"{var}_global_mean"
            ds = ds.assign({new_name: wm[var]})
            ds[new_name] = ds[new_name].assign_attrs(ds[var].attrs)
        else:
            raise ValueError("Variable for global mean calculations not in dataset.")

    if "land_sea_mask" in mask_names:

        logger.info(f"Computing domain means.")

        ds_land = mask_to_surface_type(
            ds.merge(weights), "land", surface_type_var="land_sea_mask"
        )
        weights_land = ds_land["area"]
        wm_land = _weighted_mean(ds_land, weights_land)

        ds_sea = mask_to_surface_type(
            ds.merge(weights), "sea", surface_type_var="land_sea_mask"
        )
        weights_sea = ds_sea["area"]
        wm_sea = _weighted_mean(ds_sea, weights_sea)

        for var in varnames:
            if var in ds:
                land_new_name = f"{var}_land_mean"
                sea_new_name = f"{var}_sea_mean"
                ds = ds.assign({land_new_name: wm_land[var], sea_new_name: wm_sea[var]})
                ds[land_new_name] = ds[land_new_name].assign_attrs(ds[var].attrs)
                ds[sea_new_name] = ds[sea_new_name].assign_attrs(ds[var].attrs)
            else:
                raise ValueError(
                    "Variable for land/sea mean calculations not in dataset."
                )

    if "net_precipitation_physics" in mask_names:

        logger.info(f"Computing P-E means.")

        ds_pos_PminusE = _mask_to_PminusE_sign(
            ds, "positive", "net_precipitation_physics"
        )
        weights_pos_PminusE = ds_pos_PminusE["area"]
        wm_pos_PminusE = _weighted_mean(ds_pos_PminusE, weights_pos_PminusE)

        ds_neg_PminusE = _mask_to_PminusE_sign(
            ds, "negative", "net_precipitation_physics"
        )
        weights_neg_PminusE = ds_neg_PminusE["area"]
        wm_neg_PminusE = _weighted_mean(ds_neg_PminusE, weights_neg_PminusE)

        for var in varnames:
            if var in ds:
                pos_new_name = f"{var}_pos_PminusE_mean"
                neg_new_name = f"{var}_neg_PminusE_mean"
                ds = ds.assign(
                    {
                        pos_new_name: wm_pos_PminusE[var],
                        neg_new_name: wm_neg_PminusE[var],
                    }
                )
                ds[pos_new_name] = ds[pos_new_name].assign_attrs(ds[var].attrs)
                ds[neg_new_name] = ds[neg_new_name].assign_attrs(ds[var].attrs)
            else:
                raise ValueError(
                    "Variable for sign(P - E) mean calculations not in dataset."
                )

    if "net_precipitation_physics" in mask_names and "land_sea_mask" in mask_names:

        logger.info(f"Computing domain + P-E means.")

        ds_pos_PminusE_land = mask_to_surface_type(
            ds_pos_PminusE.merge(weights), "land", surface_type_var="land_sea_mask"
        )
        weights_pos_PminusE_land = ds_pos_PminusE_land["area"]
        wm_pos_PminusE_land = _weighted_mean(
            ds_pos_PminusE_land, weights_pos_PminusE_land
        )

        ds_pos_PminusE_sea = mask_to_surface_type(
            ds_pos_PminusE.merge(weights), "sea", surface_type_var="land_sea_mask"
        )
        weights_pos_PminusE_sea = ds_pos_PminusE_sea["area"]
        wm_pos_PminusE_sea = _weighted_mean(ds_pos_PminusE_sea, weights_pos_PminusE_sea)

        ds_neg_PminusE_land = mask_to_surface_type(
            ds_neg_PminusE.merge(weights), "land", surface_type_var="land_sea_mask"
        )
        weights_neg_PminusE_land = ds_neg_PminusE_land["area"]
        wm_neg_PminusE_land = _weighted_mean(
            ds_neg_PminusE_land, weights_neg_PminusE_land
        )

        ds_neg_PminusE_sea = mask_to_surface_type(
            ds_neg_PminusE.merge(weights), "sea", surface_type_var="land_sea_mask"
        )
        weights_neg_PminusE_sea = ds_neg_PminusE_sea["area"]
        wm_neg_PminusE_sea = _weighted_mean(ds_neg_PminusE_sea, weights_neg_PminusE_sea)

        for var in varnames:
            if "z" in ds[var].dims:
                pos_land_new_name = f"{var}_pos_PminusE_land_mean"
                neg_land_new_name = f"{var}_neg_PminusE_land_mean"
                pos_sea_new_name = f"{var}_pos_PminusE_sea_mean"
                neg_sea_new_name = f"{var}_neg_PminusE_sea_mean"
                ds = ds.assign(
                    {
                        pos_land_new_name: wm_pos_PminusE_land[var],
                        neg_land_new_name: wm_neg_PminusE_land[var],
                        pos_sea_new_name: wm_pos_PminusE_sea[var],
                        neg_sea_new_name: wm_neg_PminusE_sea[var],
                    }
                )
                ds[pos_land_new_name] = ds[pos_land_new_name].assign_attrs(
                    ds[var].attrs
                )
                ds[neg_land_new_name] = ds[neg_land_new_name].assign_attrs(
                    ds[var].attrs
                )
                ds[pos_sea_new_name] = ds[pos_sea_new_name].assign_attrs(ds[var].attrs)
                ds[neg_sea_new_name] = ds[neg_sea_new_name].assign_attrs(ds[var].attrs)

    if any(
        [
            mask not in ["land_sea_mask", "net_precipitation_physics"]
            for mask in mask_names
        ]
    ):
        raise ValueError(
            'Only "land_sea_mask" and "net_precipitation_physics" are '
            "suppored as masks."
        )

    return ds


def shrink_ds(ds: xr.Dataset, config: Mapping):
    """SHrink the datast to the variables actually used in plotting
    """

    keepvars = _keepvars(config)
    dropvars = set(ds.data_vars).difference(keepvars)

    return ds.drop_vars(dropvars)


def _keepvars(config: Mapping) -> set:
    """Determine final variables in netcdf based on config
    """

    keepvars = set(
        [f"{var}_global_mean" for var in list(config["GLOBAL_MEAN_2D_VARS"])]
        + [
            f"{var}_{composite}_mean"
            for var in list(config["GLOBAL_MEAN_3D_VARS"])
            for composite in ["global", "sea", "land"]
        ]
        + [
            f"{var}_{domain}"
            for var in config["DIURNAL_VAR_MAPPING"]
            for domain in ["land", "sea", "global"]
        ]
        + [
            item
            for spec in config["DQ_MAPPING"].values()
            for item in [f"{spec['physics_name']}_physics", spec["tendency_diff_name"]]
        ]
        + [
            f"{dq_var}_{composite}"
            for dq_var in list(config["DQ_PROFILE_MAPPING"])
            for composite in list(config["PROFILE_COMPOSITES"])
        ]
        + list(config["GLOBAL_2D_MAPS"])
        + config["GRID_VARS"]
    )

    return keepvars
