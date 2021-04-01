import argparse
from copy import copy
import fsspec
import logging
import json
import numpy as np
import os
import random
import sys
from tempfile import NamedTemporaryFile
import xarray as xr
import yaml
from typing import Mapping, Sequence, Tuple, List
from toolz import dissoc

import diagnostics_utils as utils
import loaders
from vcm import safe, interpolate_to_pressure_levels
import vcm
from vcm.cloud import get_fs
import fv3fit
from ._plot_jacobian import plot_jacobian
from ._metrics import compute_metrics
from ._mapper import PredictionMapper
from ._helpers import (
    load_grid_info,
    sample_outside_train_range,
)
from ._select import meridional_transect, nearest_time


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("offline_diags")


# variables that are needed in addition to the model features
ADDITIONAL_VARS = ["pressure_thickness_of_atmospheric_layer", "pQ1", "pQ2"]
DIAGS_NC_NAME = "offline_diagnostics.nc"
DIURNAL_VARS = [
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_pQ1",
    "column_integrated_pQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]
DIURNAL_NC_NAME = "diurnal_cycle.nc"
TRANSECT_NC_NAME = "transect_lon0.nc"
METRICS_JSON_NAME = "scalar_metrics.json"
DATASET_DIM_NAME = "dataset"

# Base set of variables for which to compute column integrals and composite means
# Additional output variables are also computed.
DIAGNOSTIC_VARS = ("dQ1", "pQ1", "dQ2", "pQ2", "Q1", "Q2")
METRIC_VARS = ("dQ1", "dQ2", "Q1", "Q2")


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path", type=str, help=("Local or remote path for reading ML model."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Local or remote path where diagnostic output will be written.",
    )
    parser.add_argument(
        "--data-path",
        nargs="*",
        type=str,
        default=None,
        help=(
            "Location of test data. If not provided, will use the data_path saved "
            "with the trained model config file."
        ),
    )

    parser.add_argument(
        "--config-yml",
        type=str,
        default=None,
        help=("Config file with dataset and variable specifications."),
    )
    parser.add_argument(
        "--timesteps-file",
        type=str,
        default=None,
        help=(
            "Json file that defines train timestep set. Overrides any timestep set "
            "in training config if both are provided."
        ),
    )
    parser.add_argument(
        "--snapshot-time",
        type=str,
        default=None,
        help=(
            "Timestep to use for snapshot. Provide a string 'YYYYMMDD.HHMMSS'. "
            "If provided, will use the closest timestep in the test set. If not, will "
            "default to use the first timestep available."
        ),
    )
    parser.add_argument(
        "--timesteps-n-samples",
        type=int,
        default=None,
        help=(
            "If specified, will draw attempt to draw this many test timesteps from "
            "either i) the mapper keys that lie outside the range of times in the "
            "config timesteps or ii) the set of timesteps provided in --timesteps-file."
            "Random seed for sampling is fixed to 0. "
            "If there are not enough timesteps available outside the config range, "
            "will return all timesteps outside the range. "
            "Useful if args.config_yml is taken directly from the trained model."
            "Incompatible with also providing a timesteps-file arg. "
        ),
    )
    parser.add_argument(
        "--training",
        action="store_true",
        default=False,
        help=(
            "If provided, allows the use of timesteps from the trained model config "
            "to be used for offline diags. Only relevant if no config file is provided "
            "and no optional args for timesteps-file or timesteps-n-samples given. "
            "Acts as a safety to prevent accidental use of training set for the "
            "offline metrics."
        ),
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help=(
            "Optional path to grid data netcdf. If not provided, defaults to loading "
            "the grid  with the appropriate resolution (given in batch_kwargs) from "
            "the catalog. Useful if you do not have permissions to access the GCS "
            "data in vcm.catalog."
        ),
    )
    return parser.parse_args()


def _write_nc(ds: xr.Dataset, output_dir: str, output_file: str):
    output_file = os.path.join(output_dir, output_file)

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        get_fs(output_dir).put(tmpfile.name, output_file)
    logger.info(f"Writing netcdf to {output_file}")


def _average_metrics_dict(ds_metrics: xr.Dataset) -> Mapping:
    logger.info("Calculating metrics mean and stddev over batches...")
    metrics = {
        var: {
            "mean": float(np.mean(ds_metrics[var].values)),
            "std": float(np.std(ds_metrics[var].values)),
        }
        for var in ds_metrics.data_vars
    }
    return metrics


def _compute_diurnal_cycle(ds: xr.Dataset) -> xr.Dataset:
    return utils.create_diurnal_cycle_dataset(
        ds, ds["lon"], ds["land_sea_mask"], DIURNAL_VARS,
    )


def _compute_summary(ds: xr.Dataset, variables) -> xr.Dataset:
    # ...reduce to diagnostic variables
    summary = utils.reduce_to_diagnostic(
        ds,
        ds,
        net_precipitation=-ds["column_integrated_Q2"].sel(  # type: ignore
            derivation="target"
        ),
        primary_vars=variables,
    )
    return summary


def _fill_empty_dQ1_dQ2(ds: xr.Dataset, predicted_vars: Sequence[str]):
    template_vars = [var for var in predicted_vars if "z" in ds[var].dims]
    fill_template = ds[template_vars[0]]
    for tendency in ["dQ1", "dQ2"]:
        if tendency not in ds.data_vars:
            ds[tendency] = xr.zeros_like(fill_template)
    return ds


def _compute_diagnostics(
    batches: Sequence[xr.Dataset], grid: xr.Dataset, predicted_vars: List[str]
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    batches_summary, batches_diurnal, batches_metrics = [], [], []
    diagnostic_vars = list(
        set(list(predicted_vars) + ["dQ1", "dQ2", "pQ1", "pQ2", "Q1", "Q2"])
    )
    #diagnostic_vars = ["dQ1", "dQ2", "pQ1", "pQ2", "Q1", "Q2"]
    metric_vars = copy(predicted_vars)
    if "dQ1" in predicted_vars and "dQ2" in predicted_vars:
        metric_vars += ["Q1", "Q2"]

    # for each batch...
    for i, ds in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")
        ds = _fill_empty_dQ1_dQ2(ds, predicted_vars)

        # ...insert additional variables
        ds = (
            ds.pipe(utils.insert_total_apparent_sources)
            .pipe(utils.insert_column_integrated_vars, diagnostic_vars)
            .pipe(utils.insert_net_terms_as_Qs)
            .load()
        )
        ds.update(grid)

        diagnostic_vars_3d = [var for var in diagnostic_vars if "z" in ds[var].dims]
        ds_summary = _compute_summary(ds, diagnostic_vars_3d)
        if DATASET_DIM_NAME in ds.dims:
            sample_dims = ("time", DATASET_DIM_NAME)
        else:
            sample_dims = ("time",)  # type: ignore
        stacked = ds.stack(sample=sample_dims)

        ds_diurnal = _compute_diurnal_cycle(stacked)
        ds_metrics = compute_metrics(
            stacked,
            stacked["lat"],
            stacked["area"],
            stacked["pressure_thickness_of_atmospheric_layer"],
            predicted_vars=metric_vars,
        )

        batches_summary.append(ds_summary.load())
        batches_diurnal.append(ds_diurnal.load())
        batches_metrics.append(ds_metrics.load())
        del ds

    # then average over the batches for each output
    ds_summary = xr.concat(batches_summary, dim="batch")
    ds_diurnal = xr.concat(batches_diurnal, dim="batch").mean(dim="batch")
    ds_metrics = xr.concat(batches_metrics, dim="batch")

    ds_diagnostics, ds_scalar_metrics = _consolidate_dimensioned_data(
        ds_summary, ds_metrics
    )

    return ds_diagnostics.mean("batch"), ds_diurnal, ds_scalar_metrics


def _consolidate_dimensioned_data(ds_summary, ds_metrics):
    # moves dimensioned quantities into final diags dataset so they're saved as netcdf
    metrics_arrays_vars = [var for var in ds_metrics.data_vars if "scalar" not in var]
    ds_metrics_arrays = safe.get_variables(ds_metrics, metrics_arrays_vars)
    ds_diagnostics = ds_summary.merge(ds_metrics_arrays).rename(
        {var: var.replace("/", "-") for var in metrics_arrays_vars}
    )
    return ds_diagnostics, ds_metrics.drop(metrics_arrays_vars)


def _get_base_mapper(config: fv3fit.ModelTrainingConfig):
    logger.info("Creating base mapper")
    base_mapping_function = getattr(
        loaders.mappers, config.batch_kwargs["mapping_function"]
    )
    if not config.data_path:
        raise ValueError("Model training config has no data path attribute.")
    data_path = config.data_path
    if len(data_path) == 1:
        data_path = data_path[0]
    return base_mapping_function(
        data_path, **config.batch_kwargs.get("mapping_kwargs", {})
    )


def _get_prediction_mapper(
    config: fv3fit.ModelTrainingConfig,
    variables: Sequence[str],
    model: fv3fit.Predictor,
    grid: xr.Dataset,
):
    base_mapper = _get_base_mapper(config)
    logger.info("Creating prediction mapper")
    return PredictionMapper(base_mapper, model, grid=grid, variables=variables)


def _get_transect(ds_snapshot: xr.Dataset, grid: xr.Dataset, variables: Sequence[str]):
    ds_snapshot_regrid_pressure = xr.Dataset()
    for var in variables:
        transect_var = [
            interpolate_to_pressure_levels(
                field=ds_snapshot[var].sel(derivation=deriv),
                delp=ds_snapshot["pressure_thickness_of_atmospheric_layer"],
                dim="z",
            )
            for deriv in ["target", "predict"]
        ]
        ds_snapshot_regrid_pressure[var] = xr.concat(transect_var, dim="derivation")
    ds_snapshot_regrid_pressure = xr.merge([ds_snapshot_regrid_pressure, grid])
    ds_transect = meridional_transect(
        safe.get_variables(
            ds_snapshot_regrid_pressure, list(variables) + ["lat", "lon"]
        )
    )
    return ds_transect


def main(args):

    logger.info("Starting diagnostics routine.")

    # Safety check if user is using the training set
    if (
        not args.config_yml
        and not args.timesteps_file
        and not args.timesteps_n_samples
        and not args.training
    ):
        raise ValueError(
            "No configuration file, timesteps file, or test sample size provided. "
            "This will lead to the training set from the saved model being used "
            "to calculate offline diagnostics and metrics. "
            "If this is intended, run with the flag --training ."
        )

    if args.config_yml:
        config = fv3fit.ModelTrainingConfig.load(args.config_yml)
    else:
        config = fv3fit.load_training_config(args.model_path)
    if args.data_path:
        config.data_path = args.data_path
    config.model_path = args.model_path

    logger.info("Reading grid...")
    if not args.grid:
        # By default, read the appropriate resolution grid from vcm.catalog
        res = config.batch_kwargs.get("res", "c48")
        grid = load_grid_info(res)
    else:
        with fsspec.open(args.grid, "rb") as f:
            grid = xr.open_dataset(f).load()

    variables = list(
        set(config.input_variables + config.output_variables + ADDITIONAL_VARS)
    )

    logger.info("Opening ML model")
    model = fv3fit.load(args.model_path)
    pred_mapper = _get_prediction_mapper(config, variables, model, grid)

    # Use appropriate times if options --timesteps-file or --timesteps-n-samples given
    if args.timesteps_file:
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        if args.timesteps_n_samples:
            random.seed(0)
            timesteps = random.sample(sorted(timesteps), args.timesteps_n_samples)
        logger.info(f"Using timesteps from --timesteps-file: {timesteps}")
        config.timesteps_source = "timesteps_file"
    elif args.timesteps_n_samples:
        # Sample times outside training range and use as test set.
        train_timesteps = config.batch_kwargs.pop("timesteps", [])
        timesteps = sample_outside_train_range(
            list(pred_mapper), train_timesteps, args.timesteps_n_samples
        )
        logger.info(
            f"Using timesteps sampled from outside config timestep range: {timesteps}"
        )
        config.timesteps_source = "sampled_outside_input_config"
    else:
        try:
            timesteps = config.batch_kwargs["timesteps"]
            logger.info(f"Using timesteps given in config file: {timesteps}")
            config.timesteps_source = "input_config"
        except KeyError:
            timesteps = list(pred_mapper)
            logger.info(f"Using all timesteps available from mapper: {timesteps}")
            config.timesteps_source = "all_mapper_times"

    # Updates timesteps so that the saved config reflects the timesteps used.
    config.batch_kwargs["timesteps"] = timesteps

    # write out config used to generate diagnostics, including model path
    config.dump(args.output_path, filename="config.yaml")

    batch_kwargs = dissoc(
        config.batch_kwargs, "mapping_function", "mapping_kwargs", "timesteps"
    )
    batches = loaders.batches.batches_from_mapper(
        pred_mapper, variables, timesteps=timesteps, training=False, **batch_kwargs,
    )
    # compute diags
    ds_diagnostics, ds_diurnal, ds_scalar_metrics = _compute_diagnostics(
        batches, grid, predicted_vars=config.output_variables
    )

    # Save metadata
    cftimes = [vcm.parse_datetime_from_str(time) for time in timesteps]
    times_used = xr.DataArray(
        cftimes, dims=["time"], attrs=dict(description="times used for anaysis")
    )
    ds_diagnostics["time"] = times_used
    ds_diurnal["time"] = times_used

    # save jacobian
    try:
        plot_jacobian(model, args.output_path)  # type: ignore
    except AttributeError:
        pass

    # compute transected and zonal diags
    snapshot_time = args.snapshot_time or sorted(timesteps)[0]
    snapshot_key = nearest_time(snapshot_time, list(pred_mapper.keys()))
    ds_snapshot = pred_mapper[snapshot_key]
    transect_vertical_vars = [var for var in config.output_variables if "z" in ds_snapshot[var].dims]
    ds_transect = _get_transect(ds_snapshot, grid, transect_vertical_vars)

    # write diags and diurnal datasets
    _write_nc(ds_transect, args.output_path, TRANSECT_NC_NAME)
    _write_nc(
        ds_diagnostics, args.output_path, DIAGS_NC_NAME,
    )
    _write_nc(ds_diurnal, args.output_path, DIURNAL_NC_NAME)

    # convert and output metrics json
    metrics = _average_metrics_dict(ds_scalar_metrics)
    with fsspec.open(os.path.join(args.output_path, METRICS_JSON_NAME), "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Finished processing dataset diagnostics and metrics.")


if __name__ == "__main__":
    args = _create_arg_parser()
    main(args)
