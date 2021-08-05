import fsspec
import json
import numpy as np
import os
import random
import shutil
from typing import Mapping, Sequence, Dict, Tuple, Iterable
import warnings
import vcm
import xarray as xr

from vcm import safe
from vcm.cloud import gsutil
from vcm.catalog import catalog


UNITS = {
    "column_integrated_dq1": "[W/m2]",
    "column_integrated_dq2": "[mm/day]",
    "column_integrated_q1": "[W/m2]",
    "column_integrated_q2": "[mm/day]",
    "column_integrated_dqu": "[Pa]",
    "column_integrated_dqv": "[Pa]",
    "dq1": "[K/s]",
    "pq1": "[K/s]",
    "q1": "[K/s]",
    "dq2": "[kg/kg/s]",
    "pq2": "[kg/kg/s]",
    "q2": "[kg/kg/s]",
    "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface": "[W/m2]",
    "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface": "[W/m2]",
    "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface": "[W/m2]",
}

GRID_INFO_VARS = [
    "eastward_wind_u_coeff",
    "eastward_wind_v_coeff",
    "northward_wind_u_coeff",
    "northward_wind_v_coeff",
    "lat",
    "lon",
    "latb",
    "lonb",
    "land_sea_mask",
    "area",
]
ScalarMetrics = Dict[str, Mapping[str, float]]


def _count_features_2d(
    quantity_names: Iterable[str], dataset: xr.Dataset, sample_dim_name: str
) -> Mapping[str, int]:
    """
    count features for (sample[, z]) arrays.
    Copied from fv3fit._shared.packer, as this logic is pretty robust.
    """
    for name in quantity_names:
        if len(dataset[name].dims) > 2:
            value = dataset[name]
            raise ValueError(
                "can only count 1D/2D (sample[, z]) "
                f"variables, recieved values for {name} with dimensions {value.dims}"
            )
    return_dict = {}
    for name in quantity_names:
        value = dataset[name]
        if len(value.dims) == 1 and value.dims[0] == sample_dim_name:
            return_dict[name] = 1
        elif value.dims[0] != sample_dim_name:
            raise ValueError(
                f"cannot count values for {name} whose first dimension is not the "
                f"sample dimension ({sample_dim_name}), has dims {value.dims}"
            )
        else:
            return_dict[name] = value.shape[1]
    return return_dict


def get_variable_indices(
    data: xr.Dataset, variables: Sequence[str]
) -> Mapping[str, Tuple[int, int]]:
    if "time" in data.dims:
        data = data.isel(time=0).squeeze(drop=True)
    stacked = data.stack(sample=["tile", "x", "y"])
    variable_dims = _count_features_2d(
        variables, stacked.transpose("sample", ...), "sample"
    )
    start = 0
    variable_indices = {}
    for var, var_dim in variable_dims.items():
        variable_indices[var] = (start, start + var_dim)
        start += var_dim
    return variable_indices


def drop_physics_vars(ds: xr.Dataset):
    physics_vars = [var for var in ds if "pQ" in str(var)]
    for var in physics_vars:
        ds = ds.drop(var)
    return ds


def drop_temperature_humidity_tendencies_if_not_predicted(ds: xr.Dataset):
    tendencies = ["Q1", "Q2"]
    for name in tendencies:
        # if variable is not predicted, it will be all NaN
        if name in ds and np.all(np.isnan(ds[name].sel(derivation="predict").values)):
            ds = ds.drop(name)
    return ds


def is_3d(da: xr.DataArray, vertical_dim: str = "z"):
    return vertical_dim in da.dims


def insert_scalar_metrics_r2(
    metrics: ScalarMetrics, predicted: Sequence[str],
):
    for var in predicted:
        r2_name = f"scalar/r2/{var}/predict_vs_target"
        mse = metrics[f"scalar/mse/{var}/predict_vs_target"]["mean"]
        variance = metrics[f"scalar/mse/{var}/mean_vs_target"]["mean"]
        # std is across batches
        mse_std = metrics[f"scalar/mse/{var}/predict_vs_target"]["std"]
        variance_std = metrics[f"scalar/mse/{var}/mean_vs_target"]["std"]
        r2 = 1.0 - (mse / variance)
        r2_std = r2 * np.sqrt((mse_std / mse) ** 2 + (variance_std / variance) ** 2)
        metrics[r2_name] = {"mean": r2, "std": r2_std}
    return metrics


def insert_dataset_r2(
    ds: xr.Dataset,
    mse_coord: str = "mse",
    r2_coord: str = "r2",
    predict_coord: str = "predict",
    target_coord: str = "target",
):
    mse_vars = [
        var
        for var in ds.data_vars
        if (
            str(var).endswith(f"{predict_coord}_vs_{target_coord}")
            and mse_coord in str(var)
        )
    ]
    for mse_var in mse_vars:
        name_pieces = str(mse_var).split("-")
        variance = "-".join(name_pieces[:-1] + [f"mean_vs_{target_coord}"])
        r2_var = "-".join([s if s != mse_coord else r2_coord for s in name_pieces])
        ds[r2_var] = 1.0 - ds[mse_var] / ds[variance]
    return ds


def mse_to_rmse(ds: xr.Dataset):
    # replaces MSE variables with RMSE after the weighted avg is calculated
    mse_vars = [var for var in ds.data_vars if "mse" in str(var)]
    for mse_var in mse_vars:
        rmse_var = str(mse_var).replace("mse", "rmse")
        ds[rmse_var] = np.sqrt(ds[mse_var])
    return ds.drop(mse_vars)


def load_grid_info(res: str = "c48"):
    grid = catalog[f"grid/{res}"].read()
    wind_rotation = catalog[f"wind_rotation/{res}"].read()
    land_sea_mask = catalog[f"landseamask/{res}"].read()
    grid_info = xr.merge([grid, wind_rotation, land_sea_mask])
    return safe.get_variables(grid_info, GRID_INFO_VARS).drop("tile")


def open_diagnostics_outputs(
    data_dir,
    diagnostics_nc_name: str,
    diurnal_nc_name: str,
    transect_nc_name: str,
    metrics_json_name: str,
    metadata_json_name: str,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset, dict, dict]:
    with fsspec.open(os.path.join(data_dir, diagnostics_nc_name), "rb") as f:
        ds_diags = xr.open_dataset(f).load()
    with fsspec.open(os.path.join(data_dir, diurnal_nc_name), "rb") as f:
        ds_diurnal = xr.open_dataset(f).load()
    if vcm.get_fs(transect_nc_name).exists(transect_nc_name):
        with fsspec.open(os.path.join(data_dir, transect_nc_name), "rb") as f:
            ds_transect = xr.open_dataset(f).load()
    else:
        ds_transect = xr.Dataset()
    with fsspec.open(os.path.join(data_dir, metrics_json_name), "r") as f:
        metrics = json.load(f)
    with fsspec.open(os.path.join(data_dir, metadata_json_name), "r") as f:
        metadata = json.load(f)
    return ds_diags, ds_diurnal, ds_transect, metrics, metadata


def copy_outputs(temp_dir, output_dir):
    if output_dir.startswith("gs://"):
        gsutil.copy(temp_dir, output_dir)
    else:
        shutil.copytree(temp_dir, output_dir)


def tidy_title(var: str):
    title = (
        var.strip("pressure_level")
        .strip("zonal_avg_pressure_level")
        .strip("predict_vs_target")
        .strip("-")
        .replace("-", " ")
    )
    return title[0].upper() + title[1:]


def get_metric_string(
    metrics: Mapping[str, Mapping[str, float]],
    metric_name: str,
    var: str,
    predict_coord: str = "predict",
    target_coord: str = "target",
    precision=2,
):
    value = metrics[f"scalar/{metric_name}/{var}/{predict_coord}_vs_{target_coord}"][
        "mean"
    ]
    std = metrics[f"scalar/{metric_name}/{var}/{predict_coord}_vs_{target_coord}"][
        "std"
    ]
    return f"{value:.{precision}f} +/- {std:.{precision}f}"


def column_integrated_metric_names(metrics):
    names = []
    for key in metrics:
        if key.split("/")[0] == "scalar":
            names.append(key.split("/")[2])
    return list(set(names))


def units_from_name(var):
    return UNITS.get(var.lower(), "[units unavailable]")


def _shorten_coordinate_label(coord: str):
    # shortens and formats labels that get too long to display
    # in multi panel figures
    return (
        coord.replace("_", " ")
        .replace("average", "avg")
        .replace("precipitation", "precip")
        .replace("positive", "> 0")
        .replace("negative", "< 0")
    )


def sample_outside_train_range(all: Sequence, train: Sequence, n_sample: int,) -> list:
    # Draws test samples from outside the training time range
    if len(train) == 0:
        warnings.warn(
            "Training timestep list has zero length, test set will be drawn from "
            "the full set of timesteps in mapper."
        )
        outside_train_range = all
    else:
        outside_train_range = [t for t in all if t < min(train) or t > max(train)]
    if len(outside_train_range) == 0:
        raise ValueError("There are no timesteps available outside the training range.")
    num_test = min(len(outside_train_range), n_sample)
    random.seed(0)
    return random.sample(sorted(outside_train_range), max(1, num_test))
