#!/usr/bin/env python
"""Compute and print metrics from the output of save_prognostic_run_diags.py

This functions computes a list of named scalar performance metrics, that are useful for
comparing across models.

Usage:

    metrics.py <diagnostics netCDF file>

"""
from typing import Callable, Mapping
import xarray as xr
from toolz import curry
import json

_METRICS = []
GRID_VARS = ["lon", "lat", "lonb", "latb", "area"]


def grab_diag(ds, name):
    replace_dict = {}
    for var in ds:
        match = "_" + name
        if var.endswith(match):
            replace_dict[var] = var[: -len(match)]

    if len(replace_dict) == 0:
        raise ValueError(f"No diagnostics with name {name} found.")

    return ds[list(replace_dict.keys())].rename(replace_dict)


def to_unit_quantity(val):
    return {"value": val.item(), "units": val.units}


def to_dict(ds: xr.Dataset):
    return {key: to_unit_quantity(ds[key]) for key in ds}


def prepend_to_key(d, prefix):
    return {prefix + key: val for key, val in d.items()}


@curry
def add_to_metrics(metricname: str, func: Callable[[xr.Dataset], xr.Dataset]):
    """Register a function to be used for computing metrics

    This function will be passed the diagnostics xarray dataset,
    and should return a Dataset of scalar quantities.

    See rmse_3day below for an example.

    """

    def myfunc(diags):
        metrics = func(diags)
        return prepend_to_key(to_dict(metrics), f"{metricname}/")

    _METRICS.append(myfunc)
    return func


def compute_all_metrics(diags: xr.Dataset) -> Mapping[str, float]:
    out = {}
    for metric in _METRICS:
        out.update(metric(diags))
    return out


@add_to_metrics("rmse_3day")
def rmse_3day(diags):
    rms_global = grab_diag(diags, "rms_global").drop(GRID_VARS, errors="ignore")

    rms_global_daily = rms_global.resample(time="1D").mean()

    try:
        rms_at_day_3 = rms_global_daily.isel(time=3)
    except IndexError:  # don't compute metric if run didn't make it to 3 days
        rms_at_day_3 = xr.Dataset()

    for variable in rms_at_day_3:
        try:
            orig_unit = rms_global[variable].attrs["units"]
        except KeyError:
            raise KeyError(f"{variable} does not have units")

        rms_at_day_3[variable].attrs["units"] = orig_unit
    return rms_at_day_3


@add_to_metrics("drift_3day")
def drift_3day(diags):
    averages = grab_diag(diags, "spatial_mean_dycore_global").drop(
        GRID_VARS, errors="ignore"
    )

    daily = averages.resample(time="1D").mean()

    try:
        drift = (daily.isel(time=3) - daily.isel(time=0)) / 3
    except IndexError:  # don't compute metric if run didn't make it to 3 days
        drift = xr.Dataset()

    for variable in drift:
        orig_unit = averages[variable].attrs["units"]
        drift[variable].attrs["units"] = orig_unit + "/day"
    return drift


@add_to_metrics("time_and_global_mean_bias")
def time_mean_bias(diags):
    global_mean_bias = grab_diag(diags, "mean_bias_physics_global")

    time_and_global_mean_bias = global_mean_bias.mean("time")

    for variable in global_mean_bias:
        orig_unit = global_mean_bias[variable].attrs["units"]
        time_and_global_mean_bias[variable].attrs["units"] = orig_unit
    return time_and_global_mean_bias


def register_parser(subparsers):
    parser = subparsers.add_parser("metrics", help="Compute metrics from verification diagnostics.")
    parser.add_argument("input", help="netcdf file of compute diagnostics.")
    parser.add_argument("output")
    parser.set_defaults(func=main)


def main(args):
    diags = xr.open_dataset(args.path)
    diags["time"] = diags.time - diags.time[0]
    metrics = compute_all_metrics(diags)
    # print to stdout, use pipes to save
    print(json.dumps(metrics))
