import os

import intake


def fv3_to_vcm_conventions(data):
    rename = {
        "grid_xt": "x",
        "grid_x": "x_interface",
        "grid_yt": "y",
        "grid_y": "y_interface",
    }
    return data.rename(rename)


def open_baseline_run(url, deep_convection_emulator=True):
    physics_output_url = os.path.join(url, "physics_output.zarr")
    diags_url = os.path.join(url, "diags.zarr")

    physics_output = (
        intake.open_zarr(physics_output_url, consolidated=True)
        .to_dask()
        .resample(time="3H")
        .mean()
    )
    diags = (
        intake.open_zarr(diags_url, consolidated=True)
        .to_dask()
        .resample(time="3H")
        .mean()
    )

    merged = fv3_to_vcm_conventions(physics_output).merge(diags)
    return merged
