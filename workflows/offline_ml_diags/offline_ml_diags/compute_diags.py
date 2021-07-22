import argparse
from copy import copy
import warnings
import fsspec
import logging
import json
import numpy as np
import os
import sys
from tempfile import NamedTemporaryFile
from vcm.derived_mapping import DerivedMapping
import xarray as xr
import yaml
from typing import Mapping, Sequence, Tuple, List, Hashable

import diagnostics_utils as utils
import loaders
from vcm import safe, interpolate_to_pressure_levels
import vcm
import fv3fit
from ._plot_input_sensitivity import plot_jacobian, plot_rf_feature_importance
from ._metrics import compute_metrics
from ._helpers import (
    load_grid_info,
    is_3d,
    get_variable_indices,
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
DIURNAL_NC_NAME = "diurnal_cycle.nc"
TRANSECT_NC_NAME = "transect_lon0.nc"
METRICS_JSON_NAME = "scalar_metrics.json"
DATASET_DIM_NAME = "dataset"

# Base set of variables for which to compute column integrals and composite means
# Additional output variables are also computed.
DIAGNOSTIC_VARS = ("dQ1", "pQ1", "dQ2", "pQ2", "Q1", "Q2")
METRIC_VARS = ("dQ1", "dQ2", "Q1", "Q2")

DELP = "pressure_thickness_of_atmospheric_layer"
PREDICT_COORD = "predict"
TARGET_COORD = "target"


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path", type=str, help=("Local or remote path for reading ML model."),
    )
    parser.add_argument(
        "data_yaml",
        type=str,
        default=None,
        help=("Config file with dataset specifications."),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Local or remote path where diagnostic output will be written.",
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
    parser.add_argument(
        "--grid-resolution",
        type=str,
        default="c48",
        help=(
            "Optional grid resolution used to retrieve grid from the vcm catalog "
            '(e.g. "c48"), ignored if --grid is provided'
        ),
    )
    return parser.parse_args()


def _write_nc(ds: xr.Dataset, output_dir: str, output_file: str):
    output_file = os.path.join(output_dir, output_file)

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        vcm.get_fs(output_dir).put(tmpfile.name, output_file)
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
    diurnal_vars = [
        var
        for var in ds
        if {"tile", "x", "y", "sample", "derivation"} == set(ds[var].dims)
    ]
    return utils.create_diurnal_cycle_dataset(
        ds, ds["lon"], ds["land_sea_mask"], diurnal_vars,
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


def _fill_empty_dQ1_dQ2(ds: xr.Dataset):
    dims = ["x", "y", "tile", "z", "derivation", "time"]
    coords = {
        dim: ds.coords[dim] for dim in dims
    }  # type: Mapping[Hashable, xr.DataArray]
    fill_template = xr.DataArray(0.0, dims=dims, coords=coords)
    for tendency in ["dQ1", "dQ2"]:
        if tendency not in ds.data_vars:
            ds[tendency] = fill_template
    return ds


def _compute_diagnostics(
    batches: Sequence[xr.Dataset], grid: xr.Dataset, predicted_vars: List[str]
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    batches_summary, batches_diurnal, batches_metrics = [], [], []
    diagnostic_vars = list(
        set(list(predicted_vars) + ["dQ1", "dQ2", "pQ1", "pQ2", "Q1", "Q2"])
    )

    metric_vars = copy(predicted_vars)
    if "dQ1" in predicted_vars and "dQ2" in predicted_vars:
        metric_vars += ["Q1", "Q2"]

    # for each batch...
    for i, ds in enumerate(batches):
        logger.info(f"Processing batch {i+1}/{len(batches)}")

        ds = _fill_empty_dQ1_dQ2(ds)
        # ...insert additional variables
        ds = utils.insert_total_apparent_sources(ds)
        diagnostic_vars_3d = [var for var in diagnostic_vars if is_3d(ds[var])]

        ds = (
            ds.pipe(utils.insert_column_integrated_vars, diagnostic_vars_3d)
            .pipe(utils.insert_net_terms_as_Qs)
            .load()
        )
        ds.update(grid)
        ds_summary = _compute_summary(ds, diagnostic_vars_3d)
        if DATASET_DIM_NAME in ds.dims:
            sample_dims = ("time", DATASET_DIM_NAME)
        else:
            sample_dims = ("time",)  # type: ignore
        stacked = ds.stack(sample=sample_dims)

        ds_diurnal = _compute_diurnal_cycle(stacked)
        ds_summary["time"] = ds["time"]
        ds_diurnal["time"] = ds["time"]
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


def insert_prediction(ds: xr.Dataset, ds_pred: xr.Dataset) -> xr.Dataset:
    predicted_vars = ds_pred.data_vars
    nonpredicted_vars = [var for var in ds.data_vars if var not in predicted_vars]
    ds_target = (
        safe.get_variables(ds, [var for var in predicted_vars if var in ds.data_vars])
        .expand_dims(loaders.DERIVATION_DIM)
        .assign_coords({loaders.DERIVATION_DIM: [TARGET_COORD]})
    )
    ds_pred = ds_pred.expand_dims(loaders.DERIVATION_DIM).assign_coords(
        {loaders.DERIVATION_DIM: [PREDICT_COORD]}
    )
    return xr.merge([safe.get_variables(ds, nonpredicted_vars), ds_target, ds_pred])


def _get_predict_function(predictor, variables, grid):
    def transform(ds):
        # Prioritize dataset's land_sea_mask if grid values disagree
        ds = xr.merge(
            [ds, grid], compat="override"  # type: ignore
        )
        derived_mapping = DerivedMapping(ds)

        ds_derived = xr.Dataset({})
        for key in variables:
            try:
                ds_derived[key] = derived_mapping[key]
            except KeyError as e:
                if key == DELP:
                    raise e
                elif key in ["pQ1", "pQ2", "dQ1", "dQ2"]:
                    ds_derived[key] = xr.zeros_like(derived_mapping[DELP])
                    warnings.warn(
                        f"{key} not present in data. Filling with zeros.", UserWarning
                    )
                else:
                    raise e
        ds_prediction = predictor.predict_columnwise(ds_derived, feature_dim="z")
        return insert_prediction(ds_derived, ds_prediction)

    return transform


def main(args):
    logger.info("Starting diagnostics routine.")

    with fsspec.open(args.data_yaml, "r") as f:
        as_dict = yaml.safe_load(f)
    config = loaders.BatchesLoader.from_dict(as_dict)

    logger.info("Reading grid...")
    if not args.grid:
        # By default, read the appropriate resolution grid from vcm.catalog
        grid = load_grid_info(args.grid_resolution)
    else:
        with fsspec.open(args.grid, "rb") as f:
            grid = xr.open_dataset(f).load()

    logger.info("Opening ML model")
    model = fv3fit.load(args.model_path)
    model_variables = list(set(model.input_variables + model.output_variables + [DELP]))
    all_variables = list(set(model_variables + ADDITIONAL_VARS))

    output_data_yaml = os.path.join(args.output_path, "data_config.yaml")
    with fsspec.open(args.data_yaml, "r") as f_in, fsspec.open(
        output_data_yaml, "w"
    ) as f_out:
        f_out.write(f_in.read())

    batches = config.load_batches(model_variables)
    predict_function = _get_predict_function(model, all_variables, grid)
    batches = loaders.Map(predict_function, batches)

    # compute diags
    ds_diagnostics, ds_diurnal, ds_scalar_metrics = _compute_diagnostics(
        batches, grid, predicted_vars=model.output_variables
    )

    # save model senstivity figures: jacobian (TODO: RF feature sensitivity)
    try:
        plot_jacobian(
            model,
            os.path.join(args.output_path, "model_sensitivity_figures"),  # type: ignore
        )
    except AttributeError:
        try:
            input_feature_indices = get_variable_indices(
                data=batches[0], variables=model.input_variables
            )
            plot_rf_feature_importance(
                input_feature_indices,
                model,
                os.path.join(args.output_path, "model_sensitivity_figures"),
            )
        except AttributeError:
            pass

    if isinstance(config, loaders.BatchesFromMapperConfig):
        mapper = config.load_mapper()
        # compute transected and zonal diags
        snapshot_time = args.snapshot_time or sorted(list(mapper.keys()))[0]
        snapshot_key = nearest_time(snapshot_time, list(mapper.keys()))
        ds_snapshot = predict_function(mapper[snapshot_key])
        transect_vertical_vars = [
            var for var in model.output_variables if is_3d(ds_snapshot[var])
        ]
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
