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
    rms_global = grab_diag(diags, "rms_global").drop("area")

    rms_global_daily = rms_global.resample(time="1D").mean()

    for variable in rms_global_daily:
        try:
            orig_unit = rms_global[variable].attrs["units"]
        except KeyError:
            raise KeyError(f"{variable} does not have units")

        rms_global_daily[variable].attrs["units"] = orig_unit
    return rms_global_daily.isel(time=3)


@add_to_metrics("drift_3day")
def drift_3day(diags):
    averages = grab_diag(diags, "global_avg").drop(
        ["latb", "lonb", "area"], errors="ignore"
    )

    daily = averages.resample(time="1D").mean()
    drift = (daily.isel(time=3) - daily.isel(time=0)) / 3

    for variable in drift:
        orig_unit = averages[variable].attrs["units"]
        drift[variable].attrs["units"] = orig_unit + "/day"
    return drift


if __name__ == "__main__":
    import sys

    try:
        path = sys.argv[1]
    except IndexError:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    diags = xr.open_dataset(path)
    diags["time"] = diags.time - diags.time[0]
    metrics = compute_all_metrics(diags)
    # print to stdout, use pipes to save
    print(json.dumps(metrics))
