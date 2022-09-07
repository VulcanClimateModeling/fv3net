from runtime.segmented_run.prepare_config import HighLevelConfig
import yaml
import subprocess
import tempfile
import xarray as xr
import pytest
import os

RADIATIVE_FLUXES = [
    "clear_sky_downward_longwave_flux_at_surface_python",
    "clear_sky_downward_shortwave_flux_at_surface_python",
    "clear_sky_upward_longwave_flux_at_surface_python",
    "clear_sky_upward_shortwave_flux_at_surface_python",
    "clear_sky_upward_longwave_flux_at_top_of_atmosphere_python",
    "clear_sky_upward_shortwave_flux_at_top_of_atmosphere_python",
    "total_sky_downward_longwave_flux_at_surface_python",
    "total_sky_downward_shortwave_flux_at_surface_python",
    "total_sky_upward_longwave_flux_at_surface_python",
    "total_sky_upward_shortwave_flux_at_surface_python",
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere_python",
    "total_sky_upward_longwave_flux_at_top_of_atmosphere_python",
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere_python",
    "total_sky_longwave_heating_rate_python",
    "clear_sky_longwave_heating_rate_python",
    "total_sky_shortwave_heating_rate_python",
    "clear_sky_shortwave_heating_rate_python",
]

base_config = r"""
base_version: v0.7
initial_conditions: gs://vcm-fv3config/data/initial_conditions/c12_restart_initial_conditions/v1.0
namelist:
  coupler_nml:
    minutes: 30
    current_date:
    - 2016
    - 8
    - 1
    - 0
    - 0
    - 0
  gfdl_cloud_microphysics_nml:
    fast_sat_adj: false
  gfs_physics_nml:
    fhlwr: 900.0
    fhswr: 900.0
    hybedmf: true
    satmedmf: false
  fv_core_nml:
    npx: 13
    npy: 13
    npz: 63
fortran_diagnostics:
  - name: sfc_dt_atmos.zarr
    chunks:
      time: 2
    times:
      frequency: 900
      kind: interval
    variables:
    - {module_name: dynamics, field_name: grid_lont, output_name: lon}
    - {module_name: dynamics, field_name: grid_latt, output_name: lat}
    - {module_name: dynamics, field_name: grid_lon, output_name: lonb}
    - {module_name: dynamics, field_name: grid_lat, output_name: latb}
    - {module_name: dynamics, field_name: area, output_name: area}
    - {module_name: gfs_phys, field_name: dusfci, output_name: uflx}
    - {module_name: gfs_phys, field_name: dvsfci, output_name: vflx}
    - {module_name: gfs_phys, field_name: cnvprcpb_ave, output_name: CPRATsfc}
    - {module_name: gfs_phys, field_name: totprcpb_ave, output_name: PRATEsfc}
    - {module_name: gfs_phys, field_name: DSWRF, output_name: DSWRFsfc}
    - {module_name: gfs_phys, field_name: USWRF, output_name: USWRFsfc}
    - {module_name: gfs_phys, field_name: DSWRFtoa, output_name: DSWRFtoa}
    - {module_name: gfs_phys, field_name: USWRFtoa, output_name: USWRFtoa}
    - {module_name: gfs_phys, field_name: ULWRFtoa, output_name: ULWRFtoa}
    - {module_name: gfs_phys, field_name: ULWRF, output_name: ULWRFsfc}
    - {module_name: gfs_phys, field_name: DLWRF, output_name: DLWRFsfc}
    - {module_name: gfs_phys, field_name: lhtfl_ave, output_name: LHTFLsfc}
    - {module_name: gfs_phys, field_name: shtfl_ave, output_name: SHTFLsfc}
"""  # noqa: 501


def get_fv3config():
    config = HighLevelConfig.from_dict(yaml.safe_load(base_config))
    fv3config_dict = config.to_fv3config()
    # can't call normal dump on representation of the data table without this
    fv3config_dict["diag_table"] = fv3config_dict["diag_table"].asdict()
    return fv3config_dict


def radiation_scheme_config():
    config = get_fv3config()
    config["radiation_scheme"] = {"kind": "python"}
    config["diagnostics"] = [
        {
            "name": "radiative_fluxes.zarr",
            "chunks": {"time": 2},
            "times": {"frequency": 900, "kind": "interval"},
            "variables": RADIATIVE_FLUXES,
        }
    ]
    return config


def run_native(config, rundir):
    with tempfile.NamedTemporaryFile("w") as f:
        yaml.safe_dump(config, f)
        fv3_script = "runfv3"
        subprocess.check_call([fv3_script, "create", rundir, f.name])
        subprocess.check_call([fv3_script, "run-native", f.name, rundir])


@pytest.fixture(scope="module")
def completed_rundir(tmpdir_factory):
    config = radiation_scheme_config()
    rundir = tmpdir_factory.mktemp("rundir").join("subdir")
    run_native(config, str(rundir))
    return rundir


def test_radiative_fluxes_output(completed_rundir):
    zarrpath = os.path.join(completed_rundir, "radiative_fluxes.zarr")
    ds = xr.open_zarr(zarrpath)
    for var in RADIATIVE_FLUXES:
        assert var in ds.data_vars
