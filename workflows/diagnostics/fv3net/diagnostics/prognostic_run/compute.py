#!/usr/bin/env python
# coding: utf-8
"""
This script computes diagnostics for prognostic runs.

Diagnostics are multiple dimensional curves that can be visualized to give
a more detailed look into the data underlying the metrics.

Diagnostics are loaded and saved in groupings corresponding to the source of the data
and its intended use. E.g. the "dycore" grouping includes diagnostics output by the
dynamical core of the model and saved in `atmos_dt_atmos.tile*.nc` while the "physics"
grouping contains outputs from the physics routines (`sfc_dt_atmos.tile*.nc` and
`diags.zarr`).
"""
from argparse import ArgumentParser
import sys

import datetime
import intake
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
import fsspec
from joblib import Parallel, delayed
from time import time


from typing import Optional, Mapping, MutableMapping, Union, Tuple, Sequence

import vcm

from fv3net.diagnostics.prognostic_run import load_run_data as load_diags
from fv3net.diagnostics.prognostic_run import diurnal_cycle
from fv3net.diagnostics._shared.constants import (
    DiagArg,
    HORIZONTAL_DIMS,
    COL_DRYING,
    WVP,
    HISTOGRAM_BINS,
)
from .constants import (
    GLOBAL_AVERAGE_VARS,
    GLOBAL_BIAS_VARS,
    DIURNAL_CYCLE_VARS,
    TIME_MEAN_VARS,
    RMSE_VARS,
    PRESSURE_INTERPOLATED_VARS,
)
from fv3net.diagnostics._shared.registry import Registry
from fv3net.diagnostics._shared import transform
from fv3net.artifacts.metadata import StepMetadata


import logging

logger = logging.getLogger("SaveDiags")


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        logger.info(f"{func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def _prepare_diag_dict(suffix: str, ds: xr.Dataset) -> Mapping[str, xr.DataArray]:
    """
    Take a diagnostic dataset and add a suffix to all variable names and return as dict.
    """

    diags = {}
    for variable in ds:
        lower = variable.lower()
        da = ds[variable]
        diags[f"{lower}_{suffix}"] = da

    return diags


def _merge_diag_computes(
    input_data: Mapping[str, Tuple[xr.Dataset, xr.Dataset, xr.Dataset]],
    registries: Mapping[str, Registry],
    n_jobs: int,
) -> Mapping[str, xr.DataArray]:
    # Flattens list of all computations across registries before
    # parallelizing the computation.
    merged_input_data = []
    for registry_key, (prog, verif, grid) in input_data.items():

        if len(prog) == 0:
            logger.warn(
                f"prognostic data for {registry_key} missing. "
                "Skipping computation for {registry_key}."
            )
            continue

        if len(verif) == 0:
            logger.warn(
                f"verification data for {registry_key} missing. "
                "Skipping computation for {registry_key}."
            )
            continue

        diag_arg = DiagArg(prog, verif, grid)
        merged_input_data += [
            (func_name, func, registry_key, diag_arg)
            for func_name, func in registries[registry_key].funcs.items()
        ]

    def _compute(func_name, func, key, diag_arg):
        return registries[key].load(func_name, func, diag_arg)

    computed_outputs = Parallel(n_jobs=n_jobs, verbose=True)(
        delayed(_compute)(*compute_args) for compute_args in merged_input_data
    )
    return merge_diags(computed_outputs)


def merge_diags(diags: Sequence[Tuple[str, xr.Dataset]]) -> Mapping[str, xr.DataArray]:
    out: MutableMapping[str, xr.DataArray] = {}
    for name, ds in diags:
        out.update(_prepare_diag_dict(name, ds))
    return out


# all functions added to these registries must take three xarray Datasets as inputs
# (specifically, the prognostic run output, the verification data and the grid dataset)
# and return an xarray dataset containing one or more diagnostics.
registries = {
    "2d": Registry(merge_diags),
    "3d": Registry(merge_diags),
}
# expressions not allowed in decorator calls, so need explicit variables for each here
registry_2d = registries["2d"]
registry_3d = registries["3d"]


def rms(x, y, w, dims):
    with xr.set_options(keep_attrs=True):
        return np.sqrt(((x - y) ** 2 * w).sum(dims) / w.sum(dims))


def bias(truth, prediction):
    with xr.set_options(keep_attrs=True):
        return prediction - truth


def weighted_mean(ds, weights, dims):
    with xr.set_options(keep_attrs=True):
        return (ds * weights).sum(dims) / weights.sum(dims)


def time_mean(ds: xr.Dataset, dim: str = "time") -> xr.Dataset:
    with xr.set_options(keep_attrs=True):
        result = ds.mean(dim)
    return _assign_diagnostic_time_attrs(result, ds)


def _get_time_attrs(ds: Union[xr.Dataset, xr.DataArray]) -> Optional[Mapping[str, str]]:
    if "time" in ds.coords:
        start_time = str(ds.time.values[0])
        end_time = str(ds.time.values[-1])
        return {"diagnostic_start_time": start_time, "diagnostic_end_time": end_time}
    else:
        return None


def _assign_diagnostic_time_attrs(
    diagnostics_ds: xr.Dataset, source_ds: xr.Dataset
) -> xr.Dataset:
    for variable in diagnostics_ds:
        if variable in source_ds:
            attrs = _get_time_attrs(source_ds[variable])
            diagnostics_ds[variable] = diagnostics_ds[variable].assign_attrs(attrs)
    return diagnostics_ds


def _assign_source_attrs(
    diagnostics_ds: xr.Dataset, source_ds: xr.Dataset
) -> xr.Dataset:
    """Get attrs for each variable in diagnostics_ds from corresponding in source_ds."""
    for variable in diagnostics_ds:
        if variable in source_ds:
            attrs = source_ds[variable].attrs
            diagnostics_ds[variable] = diagnostics_ds[variable].assign_attrs(attrs)
    return diagnostics_ds


@registry_2d.register("rms_global")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
@transform.apply(transform.subset_variables, RMSE_VARS)
def rms_errors(diag_arg: DiagArg):
    logger.info("Preparing rms errors")
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )
    rms_errors = rms(prognostic, verification, grid.area, dims=HORIZONTAL_DIMS)

    return rms_errors


@registry_2d.register("zonal_and_time_mean")
@transform.apply(transform.resample_time, "1H")
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
@timer
def zonal_means_2d(diag_arg: DiagArg):
    logger.info("Preparing zonal+time means (2d)")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    zonal_means = vcm.zonal_average_approximate(
        grid.lat, prognostic, lat_name="latitude"
    )
    return time_mean(zonal_means)


@registry_3d.register("pressure_level_zonal_time_mean")
@transform.apply(transform.subset_variables, PRESSURE_INTERPOLATED_VARS)
@transform.apply(transform.skip_if_3d_output_absent)
@transform.apply(transform.resample_time, "3H")
@timer
def zonal_means_3d(diag_arg: DiagArg):
    logger.info("Preparing zonal+time means (3d)")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    logger.info(f"Computing zonal+time means (3d)")
    with xr.set_options(keep_attrs=True):
        zm = vcm.zonal_average_approximate(grid.lat, prognostic, lat_name="latitude")
        zonal_means = time_mean(zm)
    return zonal_means


@registry_3d.register("pressure_level_zonal_bias")
@transform.apply(transform.subset_variables, PRESSURE_INTERPOLATED_VARS)
@transform.apply(transform.skip_if_3d_output_absent)
@transform.apply(transform.resample_time, "3H", inner_join=True)
@timer
def zonal_bias_3d(diag_arg: DiagArg):
    logger.info("Preparing zonal mean bias (3d)")
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )
    common_vars = list(set(prognostic.data_vars).intersection(verification.data_vars))

    logger.info(f"Computing zonal+time mean biases (3d) for {common_vars}")
    with xr.set_options(keep_attrs=True):
        zm_bias = vcm.zonal_average_approximate(
            grid.lat,
            bias(verification[common_vars], prognostic[common_vars]),
            lat_name="latitude",
        )
        zonal_means = time_mean(zm_bias)
    return zonal_means


@registry_2d.register("zonal_bias")
@transform.apply(transform.resample_time, "1H")
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
@timer
def zonal_and_time_mean_biases_2d(diag_arg: DiagArg):
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )
    logger.info("Preparing zonal+time mean biases (2d)")
    common_vars = list(set(prognostic.data_vars).intersection(verification.data_vars))
    zonal_means = xr.Dataset()

    logger.info("Computing zonal+time mean biases (2d)")
    zonal_mean_bias = vcm.zonal_average_approximate(
        grid.lat,
        bias(verification[common_vars], prognostic[common_vars]),
        lat_name="latitude",
    )
    zonal_means = time_mean(zonal_mean_bias).load()
    return zonal_means


@registry_2d.register("zonal_mean_value")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
@timer
def zonal_mean_hovmoller(diag_arg: DiagArg):
    logger.info(f"Preparing zonal mean values (2d)")
    prognostic, grid = diag_arg.prediction, diag_arg.grid
    zonal_means = xr.Dataset()
    logger.info(f"Computing zonal means over time (2d)")
    with xr.set_options(keep_attrs=True):
        zonal_means = vcm.zonal_average_approximate(
            grid.lat, prognostic, lat_name="latitude"
        ).load()
    return zonal_means


@registry_2d.register("zonal_mean_bias")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.daily_mean, datetime.timedelta(days=10))
@transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
@timer
def zonal_mean_bias_hovmoller(diag_arg: DiagArg):

    logger.info(f"Preparing zonal mean biases (2d)")
    prognostic, verification, grid = (
        diag_arg.prediction,
        diag_arg.verification,
        diag_arg.grid,
    )
    common_vars = list(set(prognostic.data_vars).intersection(verification.data_vars))
    logger.info(f"Computing zonal mean biases (2d) over time for {common_vars}")
    with xr.set_options(keep_attrs=True):
        zonal_means = vcm.zonal_average_approximate(
            grid.lat,
            bias(verification[common_vars], prognostic[common_vars]),
            lat_name="latitude",
        ).load()
    return zonal_means


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_2d.register(f"spatial_min_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H")
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
    def spatial_min(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing minimum for variables ({mask_type})")
        prognostic, grid = diag_arg.prediction, diag_arg.grid
        masked = prognostic.where(~grid["area"].isnull())
        with xr.set_options(keep_attrs=True):
            return masked.min(dim=HORIZONTAL_DIMS)

    @registry_2d.register(f"spatial_max_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H")
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
    def spatial_max(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing maximum for variables ({mask_type})")
        prognostic, grid = diag_arg.prediction, diag_arg.grid
        masked = prognostic.where(~grid["area"].isnull())
        with xr.set_options(keep_attrs=True):
            return masked.max(dim=HORIZONTAL_DIMS)


for mask_type in ["global", "land", "sea", "tropics"]:

    @registry_2d.register(f"spatial_mean_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H")
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_AVERAGE_VARS)
    def global_averages_2d(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing averages for 2d variables ({mask_type})")
        prognostic, grid = diag_arg.prediction, diag_arg.grid
        return weighted_mean(prognostic, grid.area, HORIZONTAL_DIMS)

    @registry_2d.register(f"mean_bias_{mask_type}")
    @transform.apply(transform.mask_area, mask_type)
    @transform.apply(transform.resample_time, "3H", inner_join=True)
    @transform.apply(transform.daily_mean, datetime.timedelta(days=10))
    @transform.apply(transform.subset_variables, GLOBAL_BIAS_VARS)
    def global_biases_2d(diag_arg: DiagArg, mask_type=mask_type):
        logger.info(f"Preparing average biases for 2d variables ({mask_type})")
        prognostic, verification, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        bias_errors = bias(verification, prognostic)
        mean_bias_errors = weighted_mean(bias_errors, grid.area, HORIZONTAL_DIMS)
        return mean_bias_errors


@registry_2d.register("time_mean_value")
@transform.apply(transform.resample_time, "1H", inner_join=True)
@transform.apply(transform.subset_variables, TIME_MEAN_VARS)
def time_means_2d(diag_arg: DiagArg):
    logger.info("Preparing time means for 2d variables")
    prognostic = diag_arg.prediction
    return time_mean(prognostic)


@registry_2d.register("time_mean_bias")
@transform.apply(transform.resample_time, "1H", inner_join=True)
@transform.apply(transform.subset_variables, TIME_MEAN_VARS)
def time_mean_biases_2d(diag_arg: DiagArg):
    logger.info("Preparing time mean biases for 2d variables")
    prognostic, verification = diag_arg.prediction, diag_arg.verification
    return time_mean(bias(verification, prognostic))


for mask_type in ["global", "land", "sea"]:

    @registry_2d.register(f"diurnal_{mask_type}")
    @transform.apply(transform.mask_to_sfc_type, mask_type)
    @transform.apply(transform.resample_time, "1H", inner_join=True)
    @transform.apply(transform.subset_variables, DIURNAL_CYCLE_VARS)
    def _diurnal_func(diag_arg: DiagArg, mask_type=mask_type) -> xr.Dataset:
        # mask_type is added as a kwarg solely to give the logging access to the info
        logger.info(
            f"Computing diurnal cycle info for physics variables with mask={mask_type}"
        )
        prognostic, verification, grid = (
            diag_arg.prediction,
            diag_arg.verification,
            diag_arg.grid,
        )
        if len(prognostic.time) == 0:
            return xr.Dataset({})
        else:
            diag = diurnal_cycle.calc_diagnostics(prognostic, verification, grid).load()
            return _assign_diagnostic_time_attrs(diag, prognostic)


@registry_2d.register("histogram")
@transform.apply(transform.resample_time, "3H", inner_join=True, method="mean")
@transform.apply(transform.subset_variables, list(HISTOGRAM_BINS.keys()))
def compute_histogram(diag_arg: DiagArg):
    logger.info("Computing histograms for physics diagnostics")
    prognostic = diag_arg.prediction
    counts = xr.Dataset()
    for varname in prognostic.data_vars:
        count, width = vcm.histogram(
            prognostic[varname], bins=HISTOGRAM_BINS[varname], density=True
        )
        counts[varname] = count
        counts[f"{varname}_bin_width"] = width
    return _assign_source_attrs(
        _assign_diagnostic_time_attrs(counts, prognostic), prognostic
    )


@registry_2d.register("hist_bias")
@transform.apply(transform.resample_time, "3H", inner_join=True, method="mean")
@transform.apply(transform.subset_variables, list(HISTOGRAM_BINS.keys()))
def compute_histogram_bias(diag_arg: DiagArg):
    logger.info("Computing histogram biases for physics diagnostics")
    prognostic, verification = diag_arg.prediction, diag_arg.verification
    counts = xr.Dataset()
    for varname in prognostic.data_vars:
        prognostic_count, _ = vcm.histogram(
            prognostic[varname], bins=HISTOGRAM_BINS[varname], density=True
        )
        verification_count, _ = vcm.histogram(
            verification[varname], bins=HISTOGRAM_BINS[varname], density=True
        )
        counts[varname] = bias(verification_count, prognostic_count)
    return _assign_source_attrs(
        _assign_diagnostic_time_attrs(counts, prognostic), prognostic
    )


@registry_2d.register("hist_2d")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.mask_to_sfc_type, "sea")
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def compute_hist_2d(diag_arg: DiagArg):
    logger.info("Computing joint histogram of water vapor path versus Q2")
    hist2d = _compute_wvp_vs_q2_histogram(diag_arg.prediction)
    return _assign_diagnostic_time_attrs(hist2d, diag_arg.prediction)


@registry_2d.register("hist2d_bias")
@transform.apply(transform.resample_time, "3H", inner_join=True)
@transform.apply(transform.mask_to_sfc_type, "sea")
@transform.apply(transform.mask_to_sfc_type, "tropics20")
def compute_hist_2d_bias(diag_arg: DiagArg):
    logger.info("Computing bias of joint histogram of water vapor path versus Q2")
    hist2d_prog = _compute_wvp_vs_q2_histogram(diag_arg.prediction)
    hist2d_verif = _compute_wvp_vs_q2_histogram(diag_arg.verification)
    name = f"{WVP}_versus_{COL_DRYING}"
    error = bias(hist2d_verif[name], hist2d_prog[name])
    hist2d_prog.update({name: error})
    return _assign_diagnostic_time_attrs(hist2d_prog, diag_arg.prediction)


def _compute_wvp_vs_q2_histogram(ds: xr.Dataset) -> xr.Dataset:
    counts = xr.Dataset()
    bins = [HISTOGRAM_BINS[WVP], HISTOGRAM_BINS[COL_DRYING]]
    counts, wvp_bins, q2_bins = vcm.histogram2d(ds[WVP], ds[COL_DRYING], bins=bins)
    return xr.Dataset(
        {
            f"{WVP}_versus_{COL_DRYING}": counts,
            f"{WVP}_bin_width": wvp_bins,
            f"{COL_DRYING}_bin_width": q2_bins,
        }
    )


def add_catalog_and_verification_arguments(parser: ArgumentParser):
    parser.add_argument("--catalog", default=vcm.catalog.catalog_path)
    verification_group = parser.add_mutually_exclusive_group()
    verification_group.add_argument(
        "--verification",
        help="Tag for simulation to use as verification data. Checks against "
        "'simulation' metadata from intake catalog.",
        default="40day_may2020",
    )
    verification_group.add_argument(
        "--verification-url",
        default="",
        type=str,
        help="URL to segmented run. "
        "If not passed then the --verification argument is used.",
    )


def register_parser(subparsers):
    parser: ArgumentParser = subparsers.add_parser(
        "save", help="Compute the prognostic run diags."
    )
    parser.add_argument("url", help="Prognostic run output location.")
    parser.add_argument("output", help="Output path including filename.")
    add_catalog_and_verification_arguments(parser)
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Parallelism for the computation of diagnostics. "
        "Defaults to using all available cores. Can set to a lower fixed value "
        "if you are often running  into read errors when multiple processes "
        "access data concurrently.",
        default=-1,
    )
    parser.set_defaults(func=main)


def get_verification(args, catalog, join_2d="outer"):
    if args.verification_url:
        return load_diags.SegmentedRun(args.verification_url, catalog, join_2d=join_2d)
    else:
        return load_diags.CatalogSimulation(args.verification, catalog, join_2d=join_2d)


def main(args):

    logging.basicConfig(level=logging.INFO)
    attrs = vars(args)
    attrs["history"] = " ".join(sys.argv)

    # begin constructing diags
    diags = {}
    catalog = intake.open_catalog(args.catalog)
    prognostic = load_diags.SegmentedRun(args.url, catalog)
    verification = get_verification(args, catalog)
    attrs["verification"] = str(verification)

    grid = load_diags.load_grid(catalog)
    input_data = load_diags.evaluation_pair_to_input_data(
        prognostic, verification, grid
    )

    computed_diags = _merge_diag_computes(input_data, registries, args.n_jobs)
    diags.update(computed_diags)

    # add grid vars
    diags = xr.Dataset(diags, attrs=attrs)
    diags = diags.merge(grid)

    logger.info("Forcing remaining computation.")
    with ProgressBar():
        diags = diags.load()

    logger.info(f"Saving data to {args.output}")
    with fsspec.open(args.output, "wb") as f:
        vcm.dump_nc(diags, f)

    StepMetadata(
        job_type="prognostic_run_diags",
        url=args.output,
        dependencies={"prognostic_run": args.url},
        args=sys.argv[1:],
    ).print_json()
