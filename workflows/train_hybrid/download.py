import fsspec
import xarray
import numpy
from datetime import timedelta
from typing_extensions import Protocol
import xpartition

from loaders.mappers._fine_resolution_budget import eddy_flux_coarse, convergence
from vcm.fv3.metadata import gfdl_to_standard


class FineResBudget(Protocol):
    area: xarray.DataArray
    delp: xarray.DataArray
    T: xarray.DataArray
    dq3dt_deep_conv_coarse: xarray.DataArray
    dq3dt_mp_coarse: xarray.DataArray
    dq3dt_pbl_coarse: xarray.DataArray
    dq3dt_shal_conv_coarse: xarray.DataArray
    dt3dt_deep_conv_coarse: xarray.DataArray
    dt3dt_lw_coarse: xarray.DataArray
    dt3dt_mp_coarse: xarray.DataArray
    dt3dt_ogwd_coarse: xarray.DataArray
    dt3dt_pbl_coarse: xarray.DataArray
    dt3dt_shal_conv_coarse: xarray.DataArray
    dt3dt_sw_coarse: xarray.DataArray
    eddy_flux_vulcan_omega_sphum: xarray.DataArray
    eddy_flux_vulcan_omega_temp: xarray.DataArray
    exposed_area: xarray.DataArray
    qv_dt_fv_sat_adj_coarse: xarray.DataArray
    qv_dt_phys_coarse: xarray.DataArray
    sphum: xarray.DataArray
    sphum_storage: xarray.DataArray
    sphum_vulcan_omega_coarse: xarray.DataArray
    t_dt_fv_sat_adj_coarse: xarray.DataArray
    t_dt_nudge_coarse: xarray.DataArray
    t_dt_phys_coarse: xarray.DataArray
    vulcan_omega_coarse: xarray.DataArray
    T_vulcan_omega_coarse: xarray.DataArray


def open_zarr(url, consolidated=False):
    mapper = fsspec.get_mapper(url)
    return xarray.open_zarr(mapper, consolidated=consolidated)


def apparent_heating(data: FineResBudget):
    eddy_flux = eddy_flux_coarse(
        data.eddy_flux_vulcan_omega_temp,
        data.T_vulcan_omega_coarse,
        data.vulcan_omega_coarse,
        # data.T gives transpose
        data["T"],
    )
    eddy_flux_convergence = convergence(eddy_flux, data.delp, dim="pfull")
    return (
        (data.t_dt_fv_sat_adj_coarse + data.t_dt_phys_coarse - +eddy_flux_convergence)
        .assign_attrs(
            units="K/s",
            long_name="apparent heating from high resolution data",
            description=(
                "Apparent heating due to physics and sub-grid-scale advection. Given "
                "by "
                "sat adjustment (dycore) + physics tendency  - eddy-flux-convergence"
            ),
        )
        .rename("Q1")
    )


def apparent_moistening(data: FineResBudget):
    eddy_flux = eddy_flux_coarse(
        data.eddy_flux_vulcan_omega_sphum,
        data.sphum_vulcan_omega_coarse,
        data.vulcan_omega_coarse,
        data.sphum,
    )
    eddy_flux_convergence = convergence(eddy_flux, data.delp, dim="pfull")
    return (
        (data.qv_dt_fv_sat_adj_coarse + data.qv_dt_phys_coarse + eddy_flux_convergence)
        .assign_attrs(
            units="kg/kg/s",
            long_name="apparent moistening from high resolution data",
            description=(
                "Apparent moistening due to physics and sub-grid-scale advection. Given "
                "by "
                "sat adjustment (dycore) + physics tendency  - eddy-flux-convergence"
            ),
        )
        .rename("Q2")
    )


fs = fsspec.filesystem("gs")
output_location = (
    "/Users/noah/data/noahb/hybrid-fine-res/2021-05-05-hybrid-training.zarr"
)

# created by this commit:
# https://github.com/VulcanClimateModeling/vcm-workflow-control/commit/3c852d0e4f8b86c4e88db9f29f0b8e484aeb77a1
# I manually consolidated the metadata with zarr.consolidate_metadata
fine_url = "gs://vcm-ml-experiments/default/2021-04-27/2020-05-27-40-day-X-SHiELD-simulation/fine-res-budget.zarr"
fine = open_zarr(fine_url, consolidated=True)
# compute apparent sources
fine["Q1"] = apparent_heating(fine)
fine["Q2"] = apparent_moistening(fine)
# shift the data to match the other time series
fine_shifted = fine.assign(time=fine.time - timedelta(minutes=7, seconds=30))

# created by this commit
# https://github.com/VulcanClimateModeling/vcm-workflow-control/commit/dd4498bcf3143d05095bf9ff4ca3f1341ba25330
nudge_url = "gs://vcm-ml-experiments/2021-04-13-n2f-c3072/3-hrly-ave-rad-precip-setting-30-min-rad-timestep-shifted-start-tke-edmf"
nudge_physics_tendencies = open_zarr(
    nudge_url + "/physics_tendencies.zarr", consolidated=True
)
nudge_state = open_zarr(nudge_url + "/state_after_timestep.zarr", consolidated=True)
nudge_tends = open_zarr(nudge_url + "/nudging_tendencies.zarr", consolidated=True)

merged = xarray.merge(
    [gfdl_to_standard(fine_shifted), nudge_state, nudge_physics_tendencies],
    join="inner",
)

# dQ1,2,u,v
# "hybrid" definitions for humidity and moisture
merged["dQ1"] = merged["Q1"] - merged["tendency_of_air_temperature_due_to_fv3_physics"]
merged["dQ2"] = (
    merged["Q2"] - merged["tendency_of_specific_humidity_due_to_fv3_physics"]
)
merged["dQu"] = nudge_tends.x_wind_tendency_due_to_nudging
merged["dQv"] = nudge_tends.y_wind_tendency_due_to_nudging


# Select the data we want to save
output = xarray.merge(
    [
        merged.dQ1,
        merged.dQ2,
        merged.dQu,
        merged.dQv,
        merged.specific_humidity,
        merged.pressure_thickness_of_atmospheric_layer,
        merged.air_temperature,
        merged.x_wind,
        merged.y_wind,
        merged.surface_geopotential,
        merged.latitude.isel(time=0),
        merged.longitude.isel(time=0),
        merged.area,
    ],
    join="inner",
).astype(numpy.float32)

print(f"Saving to {output_location}")
output.chunk({"time": 120}).to_zarr(output_location)
