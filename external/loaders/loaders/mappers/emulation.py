import intake
import os
import xarray as xr
import vcm
import fsspec
import numpy as np

from ._base import GeoMapper


class XarrayMapper(GeoMapper):
    def __init__(self, data):
        self.data = data

        times = self.data.time.values.tolist()
        time_strings = [vcm.encode_time(time) for time in times]
        self.time_lookup = dict(zip(time_strings, times))
        self.time_string_lookup = dict(zip(times, time_strings))

    def __getitem__(self, time_string):
        return self.data.sel(time=self.time_lookup[time_string])

    def keys(self):
        return self.time_lookup.keys()


def cos_zenith_angle(time, lat, lon):
    """Daskified cos zenith angle computation"""
    datasets = []
    for time in np.asarray(time).tolist():
        datasets.append(
            xr.apply_ufunc(
                vcm.cos_zenith_angle,
                time,
                lat,
                lon,
                dask="parallelized",
                output_dtypes=[lat.dtype],
            ).assign_coords(time=time)
        )
    return xr.concat(datasets, dim="time")


def open_spencer_rundir(url):
    """
    Valid for gs://vcm-ml-experiments/2020-09-16-physics-on-nudge-to-fine/
    """

    physics_url = os.path.join(url, "physics_tendency_components.zarr")
    state_url = os.path.join(url, "data.zarr")

    physics = intake.open_zarr(physics_url, consolidated=True).to_dask()
    state = intake.open_zarr(state_url, consolidated=True).to_dask()

    tendencies_standardized = physics.rename(
        {
            "grid_xt": "x",
            "grid_yt": "y",
            "grid_x": "x_interface",
            "grid_y": "y_interface",
            "pfull": "z",
            "phalf": "z_interface",
        }
    )
    return xr.merge([state, tendencies_standardized]).transpose(
        "time", "tile", "z_interface", "z", "y_interface", "y", "x_interface", "x"
    )


def open_phis(url):

    fs = fsspec.get_fs_token_paths(url)[0]
    artifacts = fs.listdir(os.path.join(url, "artifacts"))[0]
    input_ = os.path.join(artifacts["name"], "INPUT")
    restart = vcm.open_restarts("gs://" + input_)
    return (
        restart.phis.squeeze()
        .rename({"grid_xt": "x", "grid_yt": "y"})
        .drop(["tile", "y", "x"])
        .rename("surface_geopotential")
        .load()
    )


def open_baseline_emulator(
    url="gs://vcm-ml-scratch/prognostic_runs/2020-09-25-physics-on-free",
):
    # get mapper
    data = open_spencer_rundir(url)
    data["cos_zenith_angle"] = cos_zenith_angle(data.time, data.lat, data.lon)
    data["surface_geopotential"] = open_phis(url)
    data["dQ1"] = (
        # data.tendency_of_air_temperature_due_to_microphysics
        +data.tendency_of_air_temperature_due_to_deep_convection
        # + data.tendency_of_air_temperature_due_to_shallow_convection
    )
    data["dQ2"] = (
        # data.tendency_of_specific_humidity_due_to_microphysics
        +data.tendency_of_specific_humidity_due_to_deep_convection
        # + data.tendency_of_specific_humidity_due_to_shallow_convection
    )
    return XarrayMapper(data)
