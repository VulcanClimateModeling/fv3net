import xarray as xr
from os.path import join
from datetime import datetime, timedelta
import cftime
import pandas as pd


data_id = "data/raw/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute"
output_2d = "data/interim/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute_2d.zarr"
output_3d = "data/interim/2019-07-17-GFDL_FV3_DYAMOND_0.25deg_15minute_3d.zarr"

files_3d = [
    "h_plev_C3072_1536x768.fre.nc",
    "qi_plev_C3072_1536x768.fre.nc",
    "ql_plev_C3072_1536x768.fre.nc",
    "q_plev_C3072_1536x768.fre.nc",
    "t_plev_C3072_1536x768.fre.nc",
    "u_plev_C3072_1536x768.fre.nc",
    "v_plev_C3072_1536x768.fre.nc",
    "pres_C3072_1536x768.fre.nc",
    "qi_C3072_1536x768.fre.nc",
    "ql_C3072_1536x768.fre.nc",
    "qr_C3072_1536x768.fre.nc",
    "qv_C3072_1536x768.fre.nc",
    "temp_C3072_1536x768.fre.nc",
    "w_C3072_1536x768.fre.nc",
    "u_C3072_1536x768.fre.nc",
    "v_C3072_1536x768.fre.nc",
]

files_2d = [
    "cape_C3072_1536x768.fre.nc",
    "cin_C3072_1536x768.fre.nc",
    "cldc_C3072_1536x768.fre.nc",
    "flds_C3072_1536x768.fre.nc",
    "flus_C3072_1536x768.fre.nc",
    "flut_C3072_1536x768.fre.nc",
    "fsds_C3072_1536x768.fre.nc",
    "fsdt_C3072_1536x768.fre.nc",
    "fsus_C3072_1536x768.fre.nc",
    "fsut_C3072_1536x768.fre.nc",
    "h500_C3072_1536x768.fre.nc",
    "intqg_C3072_1536x768.fre.nc",
    "intqi_C3072_1536x768.fre.nc",
    "intql_C3072_1536x768.fre.nc",
    "intqr_C3072_1536x768.fre.nc",
    "intqs_C3072_1536x768.fre.nc",
    "intqv_C3072_1536x768.fre.nc",
    "lhflx_C3072_1536x768.fre.nc",
    "pr_C3072_1536x768.fre.nc",
    "ps_C3072_1536x768.fre.nc",
    "q2m_C3072_1536x768.fre.nc",
    "rh500_C3072_1536x768.fre.nc",
    "rh700_C3072_1536x768.fre.nc",
    "rh850_C3072_1536x768.fre.nc",
    "shflx_C3072_1536x768.fre.nc",
    "t2m_C3072_1536x768.fre.nc",
    "ts_C3072_1536x768.fre.nc",
    "u10m_C3072_1536x768.fre.nc",
    "u200_C3072_1536x768.fre.nc",
    "ustrs_C3072_1536x768.fre.nc",
    "v10m_C3072_1536x768.fre.nc",
    "qs_C3072_1536x768.fre.nc",
    "v200_C3072_1536x768.fre.nc",
    "vstrs_C3072_1536x768.fre.nc",
]

BOTH_DIRS = ['INPUT', 'RESTART']
CATEGORY_DIR_MAPPING = {
    'grid_spec' : ['.'],
    'oro_data' : ['INPUT'],
    'fv_core.res' : BOTH_DIRS,
    'fv_srf_wnd.res' : BOTH_DIRS,
    'fv_tracer.res' : BOTH_DIRS,
    'sfc_data' : BOTH_DIRS,
    'phy_data' : ['RESTART']
}

# GRID_SPEC_AXES_MAP = {
#     'grid_x' : 'xaxis_2',
#     'grid_y' : 'yaxis_1',
#     'grid_xt' : 'xaxis_1',
#     'grid_yt' : 'yaxis_2'
# }

ORO_DATA_AXES_MAP = {
    'lon' : 'grid_xt',
    'lat' : 'grid_yt'
}

# rename restart file dimensions according to diagnostics conventions
OUTPUT_AXES_MAP = {
    'xaxis_1' : 'grid_xt',
    'xaxis_2' : 'grid_x',
    'yaxis_1' : 'grid_yt',
    'yaxis_2' : 'grid_y',
    'zaxis_1' : 'pfull',
    'zaxis_2' : 'phalf'
}

N_TILES = 6
TILE_SUFFIXES = [f".tile{tile}.nc" for tile in range(1, N_TILES + 1)]
TIME_FMT='%Y%m%d.%H%M%S'
TIMESTEP_LENGTH_MINUTES = 15


def open_files(files, **kwargs):
    paths = [join(data_id, path) for path in files]
    return xr.open_mfdataset(paths, **kwargs)


def run_dir_cubed_sphere_filepaths(
    run_dir,
    category,
    tile_suffixes,
    target_dirs
):
    return [[
        [join(join(run_dir, target_dir), category + tile_suffix) for tile_suffix in tile_suffixes]
        for target_dir in target_dirs]]


def assign_time_dims(ds, dirs, dims):
    if dirs == BOTH_DIRS:
        ds = ds.assign_coords(initialization_time=dims['initialization_time'], forecast_time=dims['forecast_time'])
    elif dirs == ['RESTART']:
        ds = ds.assign_coords(initialization_time=dims['initialization_time'], forecast_time=[dims['forecast_time'][-1]])
    if 'Time' in ds.dims:
        ds = ds.squeeze(dim = 'Time')
    if 'Time' in ds.coords:
        ds = ds.drop(labels = 'Time')
    if 'time' in ds.coords:
        ds = ds.drop(labels = 'time')
    return ds


def open_oro_data(paths, oro_data_mapping):
    '''
    orograph files need to be opened via xr.concat since they are indexed differently than other files
    '''
    ds = xr.concat(
        objs=[xr.open_dataset(path).drop(labels=['lat', 'lon']) for path in paths[0][0]],
        dim='tile'
    )
    ds = ds.rename(oro_data_mapping)
    return ds.drop(labels='slmsk')


def open_fv_tracer(paths):
    '''
    This file type needs to be handled differently than the others 
    because its input and ouput variable lists are currently not the same -
    sgs_tke shows up in outputs but not inputs
    '''
    input_ds = xr.open_mfdataset(paths=paths[0][0], concat_dim = ['tile'], combine='nested').drop(labels='sgs_tke')
    restart_ds = xr.open_mfdataset(paths=paths[0][1], concat_dim = ['tile'], combine='nested')
    return xr.concat([input_ds, restart_ds], dim='forecast_time').expand_dims(dim='initialization_time')


def add_vertical_coords(ds, run_dir, dims):
    vertical_coords_ds = xr.open_dataset(join(run_dir, 'INPUT/fv_core.res.nc')).rename({'xaxis_1' : 'phalf'}).squeeze().drop(labels='Time')
    return xr.merge([ds, vertical_coords_ds])


def use_diagnostic_coordinates(ds, category, output_mapping):
    '''
    Map the coordinate names to diagnostic standards
    Note that a special fix is required since the dimension names are not consistent across file categories
    '''
    for coord in ds.coords:
        if coord in output_mapping:
            if not (category == 'fv_core.res' and coord.startswith('y')):
                ds = ds.rename({coord : output_mapping[coord]})
            else:
                if coord == 'yaxis_1':
                    ds = ds.rename({coord : 'grid_y'})
                if coord == 'yaxis_2':
                    ds = ds.rename({coord : 'grid_yt'})
    return ds


def combine_file_categories(
    run_dir,
    category_mapping,
    tile_suffixes,
    new_dims,
    oro_data_mapping,
    output_mapping
):
    ds_dict = {}
    for category, dirs in category_mapping.items():
        paths = run_dir_cubed_sphere_filepaths(
            run_dir=run_dir,
            category=category,
            tile_suffixes=tile_suffixes,
            target_dirs=dirs
        )
        if category == 'oro_data':
            ds = open_oro_data(paths, oro_data_mapping)
        elif category == 'fv_tracer.res':
            ds = open_fv_tracer(paths)
        else:
            ds = xr.open_mfdataset(
                paths=paths,
                concat_dim=['initialization_time', 'forecast_time', 'tile'],
                combine='nested'
            )
        if 'tile' in new_dims:
            ds = ds.assign_coords(tile=new_dims['tile'])
        if category == 'grid_spec':
            # remove time dim from grid spec
            ds = ds.squeeze()
        if category == 'sfc_data':
            # give LSM soil layers their own dimension so as not to confuse them with the atmosphere
            ds = ds.rename({'zaxis_1' : 'soil_layers'})
        ds = assign_time_dims(ds, dirs, new_dims)
        ds_dict[category] = use_diagnostic_coordinates(ds, category, output_mapping)
    ds_merged = add_vertical_coords(ds=xr.merge([ds for ds in ds_dict.values()]), run_dir=run_dir, dims=new_dims)
    return ds_merged


def rundir_to_dataset(rundir: str, initial_timestep: str) -> xr.Dataset:
    tile = pd.Index(range(1, N_TILES + 1), name='tile')
    t = datetime.strptime(initial_timestep, TIME_FMT)
    initialization_time = [cftime.DatetimeJulian(
        t.year,
        t.month,
        t.day,
        t.hour,
        t.minute,
        t.second)]
    # TODO: add functionality for passing timestep length and number of restart timesteps
    forecast_time = [timedelta(minutes = 0), timedelta(minutes = TIMESTEP_LENGTH_MINUTES)]
    new_dims = {'initialization_time' : initialization_time, 'forecast_time' : forecast_time, 'tile' : tile}
    ds = combine_file_categories(
        run_dir=rundir,
        category_mapping=CATEGORY_DIR_MAPPING,
        new_dims=new_dims,
        output_mapping=OUTPUT_AXES_MAP,
        oro_data_mapping=ORO_DATA_AXES_MAP,
        tile_suffixes=TILE_SUFFIXES
    )
    return ds


def main():
    ds_3d = open_files(files_3d)
    ds_3d.to_zarr(output_3d, mode="w")

    ds_2d = open_files(files_2d)
    ds_2d.to_zarr(output_2d, mode="w")


if __name__ == "__main__":
    main()
