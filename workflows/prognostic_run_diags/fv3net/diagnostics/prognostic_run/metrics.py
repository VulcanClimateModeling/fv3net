#!/usr/bin/env python
"""Compute and print metrics from the output of save_prognostic_run_diags.py

This functions computes a list of named scalar performance metrics, that are useful for
comparing across models.

Usage:

    metrics.py <diagnostics netCDF file>

"""
from typing import Callable, Mapping
import numpy as np
import xarray as xr
from toolz import curry
from .constants import HORIZONTAL_DIMS, PERCENTILES
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


def weighted_mean(ds, w, dims):
    return (ds * w).sum(dims) / w.sum(dims)


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

    restore_units(rms_global, rms_at_day_3)
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


@add_to_metrics("time_and_global_mean_value")
def time_and_global_mean_value(diags):
    time_mean_value = grab_diag(diags, "time_mean_value")
    area = diags["area"]
    time_and_global_mean_value = weighted_mean(time_mean_value, area, HORIZONTAL_DIMS)
    restore_units(time_mean_value, time_and_global_mean_value)
    return time_and_global_mean_value


@add_to_metrics("time_and_global_mean_bias")
def time_and_global_mean_bias(diags):
    time_mean_bias = grab_diag(diags, "time_mean_bias")
    area = diags["area"]
    time_and_global_mean_bias = weighted_mean(time_mean_bias, area, HORIZONTAL_DIMS)
    restore_units(time_mean_bias, time_and_global_mean_bias)
    return time_and_global_mean_bias


@add_to_metrics("rmse_of_time_mean")
def rmse_time_mean(diags):
    time_mean_bias = grab_diag(diags, "time_mean_bias")
    area = diags["area"]
    rms_of_time_mean_bias = np.sqrt(
        weighted_mean(time_mean_bias ** 2, area, HORIZONTAL_DIMS)
    )
    restore_units(time_mean_bias, rms_of_time_mean_bias)
    return rms_of_time_mean_bias


for percentile in PERCENTILES:

    @add_to_metrics(f"percentile_{percentile}")
    def percentile_metric(diags, percentile=percentile):
        histogram = grab_diag(diags, "histogram")
        percentiles = xr.Dataset()
        data_vars = [v for v in histogram.data_vars if not v.endswith("bin_width")]
        for varname in data_vars:
            percentiles[varname] = compute_percentile(
                percentile,
                histogram[varname].values,
                histogram[f"{varname}_bins"].values,
                histogram[f"{varname}_bin_width"].values,
            )
        restore_units(histogram, percentiles)
        return percentiles


def compute_percentile(
    percentile: float, freq: np.ndarray, bins: np.ndarray, bin_widths: np.ndarray
) -> float:
    """Compute percentile given normalized histogram.

    Args:
        percentile: value between 0 and 100
        freq: array of frequencies normalized by bin widths
        bins: values of left sides of bins
        bin_widths: values of bin widths

    Returns:
        value of distribution at percentile
    """
    cumulative_distribution = np.cumsum(freq * bin_widths)
    if np.abs(cumulative_distribution[-1] - 1) > 1e-6:
        raise ValueError(
            "The provided frequencies do not integrate to one. "
            "Ensure that histogram is computed with density=True."
        )
    bin_midpoints = bins + 0.5 * bin_widths
    closest_index = np.argmin(np.abs(cumulative_distribution - percentile / 100))
    return bin_midpoints[closest_index]


def restore_units(source, target):
    for variable in target:
        target[variable].attrs["units"] = source[variable].attrs["units"]


def register_parser(subparsers):
    parser = subparsers.add_parser(
        "metrics",
        help="Compute metrics from verification diagnostics. "
        "Prints to standard output.",
    )
    parser.add_argument("input", help="netcdf file of prognostic_run_diags save.")
    parser.set_defaults(func=main)


def main(args):
    diags = xr.open_dataset(args.input)
    diags["time"] = diags.time - diags.time[0]
    metrics = compute_all_metrics(diags)
    # print to stdout, use pipes to save
    print(json.dumps(metrics))
