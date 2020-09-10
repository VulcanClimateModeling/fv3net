import argparse
from datetime import datetime
import fsspec
import intake
import json
import logging
import os
import pandas as pd
import tempfile
from typing import Sequence
import xarray as xr
import zarr.storage as zstore

from diagnostics_utils.region import RegionOfInterest, equatorial_zone
from loaders.mappers import open_fine_res_apparent_sources
import _utils as utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vertical_profile_plots")
xr.set_options(keep_attrs=True)

TIME_FMT = "%Y%m%d.%H%M%S"
RENAME_VARS = {
    "grid_xt": "x",
    "grid_x": "x_interface",
    "grid_yt": "y",
    "grid_y": "y_interface",
    "pfull": "z",
    "delp": "pressure_thickness_of_atmospheric_layer",
    "temp": "air_temperature",
    "sphum": "specific_humidity",
    "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
    "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
}
DATA_VARS = [
    "pressure_thickness_of_atmospheric_layer",
     "air_temperature",
     "specific_humidity", 
     "pQ1",
     "pQ2"
]

def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_data_path",
        type=str,
        help="Location of run data."
    )
    parser.add_argument(
        "fine_res_reference_path",
        type=str,
        help="Location of reference fine res data."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    parser.add_argument(
        "--lat-bounds",
        nargs=2,
        type=float,
        help=(
            "Min, max latitude bounds"
        ),
    )
    parser.add_argument(
        "--lon-bounds",
        nargs=2,
        type=float,
        help=(
            "Min, max longitude bounds"
        ),
    )
    parser.add_argument(
        "--consolidated",
        type=bool,
        default=False,
        help="Is zarr metadata consolidated?"
    )
    parser.add_argument(
        "--time-bounds",
        nargs=2,
        type=str,
        help="Optional, min/max time range. Should have format 'YYYYMMDD.HHMMSS'."
    )
    parser.add_argument(
        "--mapper-function",
        type=str,
        help="Optional, provide if reading vertical profiles from training data."
    )
    parser.add_argument(
        "--mapper-kwargs",
        type=json.loads,
        help="Optional, use if using a mapper to read training data."
    )
    parser.add_argument(
        "--catalog-path",
        type=str,
        default="catalog.yml",
        help="Path to catalog from where script is executed"
    )
    return parser.parse_args()


def _open_zarr(
    url: str, time_bounds: Sequence[str] = None, consolidated: bool = False
) :
    mapper = fsspec.get_mapper(url)
    if time_bounds:
        time_slice = slice(*[datetime.strptime(t, TIME_FMT) for t in time_bounds])
    else:
        time_slice = slice(None, None)
    ds = xr.open_zarr(
        zstore.LRUStoreCache(mapper, 1024),
        consolidated=consolidated,
        mask_and_scale=False,
    ) 
    renamed = {
        key: value for key, value in RENAME_VARS.items()
        if key in ds.data_vars}
    ds = ds.rename(renamed)[DATA_VARS] \
        .pipe(utils.standardize_zarr_time_coord) \
        .sel({"time": time_slice})
    return ds


def _fine_res_reference(
    fine_res_path: str,
    times: Sequence[datetime]
):
    mapper = open_fine_res_apparent_sources(fine_res_path, offset_seconds=450)
    times = [pd.to_datetime(t).strftime(TIME_FMT) for t in times]
    return utils.dataset_from_timesteps(
        mapper, times, ["air_temperature", "specific_humidity"])


if __name__ == "__main__":
    args = _create_arg_parser()

    cat = intake.open_catalog(args.catalog_path)
    grid = cat["grid/c48"].to_dask()

    if ".zarr" in args.run_data_path:
        ds = _open_zarr(args.run_data_path, args.time_bounds, args.consolidated,)
    ds = xr.merge([ds, grid])
    fine_res = _fine_res_reference(args.fine_res_reference_path, ds.time.values)
    for var in ["air_temperature", "specific_humidity"]:
        ds[f"{var}_anomaly"] = ds[var] - fine_res[var]

    if args.lat_bounds and args.lon_bounds:
        region = RegionOfInterest(tuple(args.lat_bounds), tuple(args.lon_bounds))
    else:
        region = equatorial_zone
    ds = region.average(ds)
    with tempfile.TemporaryDirectory() as tmpdir:
        for var in [
        "air_temperature_anomaly",
        "specific_humidity_anomaly", 
        "pQ1", 
        "pQ2", ]:
            fig = utils.time_series(ds[var], grid)
            outfile = os.path.join(tmpdir, f"{var}_profile_time_series.png")
            fig.savefig(outfile)
            logger.info("Saved figure {var} vertical profile.")

        utils.copy_outputs(tmpdir, args.output_dir)