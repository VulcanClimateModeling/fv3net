import os
import pytest
import yaml
import subprocess
from datetime import time, timedelta, datetime
from pathlib import Path

import fv3config

# need to check if fv3gfs exists in a subprocess, importing fv3gfs into this module
# causes tests to fail. Not sure why.
# See https://github.com/VulcanClimateModeling/fv3gfs-python/issues/79
# - noah
FV3GFS_INSTALLED = subprocess.call(["python", "-c", "import fv3gfs"]) == 0
with_fv3gfs = pytest.mark.skipif(not FV3GFS_INSTALLED, reason="fv3gfs not installed")

PREP_CONFIG_PY = Path(__file__).parent.parent.joinpath("nudging/prepare_config.py").as_posix()
RUNFILE_PY = Path(__file__).parent.parent.joinpath("nudging/runfile.py").as_posix()
default_config = r"""
base_version: v0.4
data_table: default
diag_table: default
experiment_name: default_experiment
forcing: ""
initial_conditions: ""
namelist:
  amip_interp_nml:
    data_set: reynolds_oi
    date_out_of_range: climo
    interp_oi_sst: true
    no_anom_sst: false
    use_ncep_ice: false
    use_ncep_sst: true
  atmos_model_nml:
    blocksize: 24
    chksum_debug: false
    dycore_only: false
    fdiag: 0.0
    fhmax: 1024.0
    fhmaxhf: -1.0
    fhout: 0.25
    fhouthf: 0.0
  cires_ugwp_nml:
    knob_ugwp_azdir:
    - 2
    - 4
    - 4
    - 4
    knob_ugwp_doaxyz: 1
    knob_ugwp_doheat: 1
    knob_ugwp_dokdis: 0
    knob_ugwp_effac:
    - 1
    - 1
    - 1
    - 1
    knob_ugwp_ndx4lh: 4
    knob_ugwp_solver: 2
    knob_ugwp_source:
    - 1
    - 1
    - 1
    - 0
    knob_ugwp_stoch:
    - 0
    - 0
    - 0
    - 0
    knob_ugwp_version: 0
    knob_ugwp_wvspec:
    - 1
    - 32
    - 32
    - 32
    launch_level: 55
  coupler_nml:
    atmos_nthreads: 1
    calendar: julian
    force_date_from_namelist: true
    current_date:
    - 2016
    - 8
    - 1
    - 0
    - 0
    - 0
    days: 0
    dt_atmos: 900
    dt_ocean: 900
    hours: 0
    memuse_verbose: true
    minutes: 15
    months: 0
    ncores_per_node: 32
    seconds: 0
    use_hyper_thread: true
  diag_manager_nml:
    prepend_date: false
  external_ic_nml:
    checker_tr: false
    filtered_terrain: true
    gfs_dwinds: true
    levp: 80
    nt_checker: 0
  fms_io_nml:
    checksum_required: false
    max_files_r: 100
    max_files_w: 100
  fms_nml:
    clock_grain: ROUTINE
    domains_stack_size: 3000000
    print_memory_usage: false
  fv_core_nml:
    a_imp: 1.0
    adjust_dry_mass: false
    beta: 0.0
    consv_am: false
    consv_te: 1.0
    d2_bg: 0.0
    d2_bg_k1: 0.16
    d2_bg_k2: 0.02
    d4_bg: 0.15
    d_con: 1.0
    d_ext: 0.0
    dddmp: 0.2
    delt_max: 0.002
    dnats: 1
    do_sat_adj: true
    do_vort_damp: true
    dwind_2d: false
    external_ic: false
    fill: true
    fv_debug: false
    fv_sg_adj: 900
    gfs_phil: false
    hord_dp: 6
    hord_mt: 6
    hord_tm: 6
    hord_tr: 8
    hord_vt: 6
    hydrostatic: false
    io_layout:
    - 1
    - 1
    k_split: 1
    ke_bg: 0.0
    kord_mt: 10
    kord_tm: -10
    kord_tr: 10
    kord_wz: 10
    layout:
    - 1
    - 1
    make_nh: false
    mountain: true
    n_split: 6
    n_sponge: 4
    na_init: 0
    ncep_ic: false
    nggps_ic: false
    no_dycore: false
    nord: 2
    npx: 13
    npy: 13
    npz: 63
    ntiles: 6
    nudge: false
    nudge_qv: true
    nwat: 6
    p_fac: 0.1
    phys_hydrostatic: false
    print_freq: 3
    range_warn: true
    reset_eta: false
    rf_cutoff: 800.0
    rf_fast: false
    tau: 5.0
    use_hydro_pressure: false
    vtdm4: 0.06
    warm_start: true
    z_tracer: true
  fv_grid_nml: {}
  gfdl_cloud_microphysics_nml:
    c_cracw: 0.8
    c_paut: 0.5
    c_pgacs: 0.01
    c_psaci: 0.05
    ccn_l: 300.0
    ccn_o: 100.0
    const_vg: false
    const_vi: false
    const_vr: false
    const_vs: false
    de_ice: false
    do_qa: true
    do_sedi_heat: false
    dw_land: 0.16
    dw_ocean: 0.1
    fast_sat_adj: true
    fix_negative: true
    icloud_f: 1
    mono_prof: true
    mp_time: 450.0
    prog_ccn: false
    qi0_crt: 8.0e-05
    qi_lim: 1.0
    ql_gen: 0.001
    ql_mlt: 0.001
    qs0_crt: 0.001
    rad_graupel: true
    rad_rain: true
    rad_snow: true
    rh_inc: 0.3
    rh_inr: 0.3
    rh_ins: 0.3
    rthresh: 1.0e-05
    sedi_transport: false
    tau_g2v: 900.0
    tau_i2s: 1000.0
    tau_l2v:
    - 225.0
    tau_v2l: 150.0
    use_ccn: true
    use_ppm: false
    vg_max: 12.0
    vi_max: 1.0
    vr_max: 12.0
    vs_max: 2.0
    z_slope_ice: true
    z_slope_liq: true
  gfs_physics_nml:
    cal_pre: false
    cdmbgwd:
    - 3.5
    - 0.25
    cnvcld: false
    cnvgwd: true
    debug: false
    dspheat: true
    fhcyc: 24.0
    fhlwr: 3600.0
    fhswr: 3600.0
    fhzero: 0.25
    hybedmf: true
    iaer: 111
    ialb: 1
    ico2: 2
    iems: 1
    imfdeepcnv: 2
    imfshalcnv: 2
    imp_physics: 11
    isol: 2
    isot: 1
    isubc_lw: 2
    isubc_sw: 2
    ivegsrc: 1
    ldiag3d: false
    lwhtr: true
    ncld: 5
    nst_anl: true
    pdfcld: false
    pre_rad: false
    prslrd0: 0.0
    random_clds: false
    redrag: true
    shal_cnv: true
    swhtr: true
    trans_trac: true
    use_ufo: true
  interpolator_nml:
    interp_method: conserve_great_circle
  nam_stochy:
    lat_s: 96
    lon_s: 192
    ntrunc: 94
  namsfc:
    fabsl: 99999
    faisl: 99999
    faiss: 99999
    fnabsc: grb/global_mxsnoalb.uariz.t1534.3072.1536.rg.grb
    fnacna: ''
    fnaisc: grb/CFSR.SEAICE.1982.2012.monthly.clim.grb
    fnalbc: grb/global_snowfree_albedo.bosu.t1534.3072.1536.rg.grb
    fnalbc2: grb/global_albedo4.1x1.grb
    fnglac: grb/global_glacier.2x2.grb
    fnmskh: grb/seaice_newland.grb
    fnmxic: grb/global_maxice.2x2.grb
    fnslpc: grb/global_slope.1x1.grb
    fnsmcc: grb/global_soilmgldas.t1534.3072.1536.grb
    fnsnoa: ''
    fnsnoc: grb/global_snoclim.1.875.grb
    fnsotc: grb/global_soiltype.statsgo.t1534.3072.1536.rg.grb
    fntg3c: grb/global_tg3clim.2.6x1.5.grb
    fntsfa: ''
    fntsfc: grb/RTGSST.1982.2012.monthly.clim.grb
    fnvegc: grb/global_vegfrac.0.144.decpercent.grb
    fnvetc: grb/global_vegtype.igbp.t1534.3072.1536.rg.grb
    fnvmnc: grb/global_shdmin.0.144x0.144.grb
    fnvmxc: grb/global_shdmax.0.144x0.144.grb
    fnzorc: igbp
    fsicl: 99999
    fsics: 99999
    fslpl: 99999
    fsmcl:
    - 99999
    - 99999
    - 99999
    fsnol: 99999
    fsnos: 99999
    fsotl: 99999
    ftsfl: 99999
    ftsfs: 90
    fvetl: 99999
    fvmnl: 99999
    fvmxl: 99999
    ldebug: false
nudging:
  restarts_path: ""
  timescale_hours:
    air_temperature: 3.0
    specific_humidity: 3.0
    x_wind: 3.0
    y_wind: 3.0
"""  # noqa


# Test two nudging timesteps
START_TIME = [2016, 8, 1, 0, 0, 0]
TIMESTEP_SECONDS = 900
RUNTIME_MINUTES = 30
TIME_FMT = "%Y%m%d.%H%M%S"
RUNTIME = {"days": 0, "months": 0, "hours": 0, "minutes": RUNTIME_MINUTES, "seconds": 0}
BASE_FV3CONFIG_CACHE = Path("/inputdata/fv3config-cache", "gs", "vcm-fv3config", "vcm-fv3config", "data")


def get_nudging_config(config_yaml: str, restart_dir: str):
    config = yaml.safe_load(config_yaml)
    coupler_nml = config["namelist"]["coupler_nml"]
    coupler_nml["current_date"] = START_TIME
    coupler_nml.update(RUNTIME)
    coupler_nml["dt_atmos"] = TIMESTEP_SECONDS
    coupler_nml["dt_ocean"] = TIMESTEP_SECONDS

    ic_path = BASE_FV3CONFIG_CACHE.joinpath("initial_conditions", "c12_restart_initial_conditions", "v1.0")
    config["initial_conditions"] = ic_path.as_posix()
    forcing_path = BASE_FV3CONFIG_CACHE.joinpath("base_forcing", "v1.1")
    config["forcing"] = forcing_path.as_posix()
    config["nudging"]["restarts_path"] = Path(restart_dir).as_posix()
    oro_path = BASE_FV3CONFIG_CACHE.joinpath("orographic_data", "v1.0")
    config["orographic_forcing"] = oro_path.as_posix()
    return config


@pytest.fixture
def tmpdir_restart_dir(tmpdir):
    
    minute_per_step = TIMESTEP_SECONDS // 60
    nudge_timesteps = RUNTIME_MINUTES // minute_per_step
    restart_path = Path(BASE_FV3CONFIG_CACHE, "initial_conditions", "c12_restart_initial_conditions", "v1.0")

    tmp_restarts = Path(tmpdir, "restarts")
    tmp_restarts.mkdir(exist_ok=True)

    start = datetime(*START_TIME)
    delta_t = timedelta(minutes=minute_per_step)
    for i in range(nudge_timesteps + 1):
        timestamp = (start + i * delta_t).strftime(TIME_FMT)

        # Make timestamped restart directory
        restart_dir = tmp_restarts.joinpath(timestamp)
        restart_dir.mkdir(parents=True, exist_ok=True)

        _symlink_restarts(restart_dir, restart_path, timestamp)

    return tmpdir, tmp_restarts.as_posix()


def _symlink_restarts(timestamp_dir: Path, target_dir: Path, symlink_prefix: str):
    files_to_glob = ["fv_core*", "fv_srf_wnd*", "fv_tracer*", "sfc_data*"]

    for file_pattern in files_to_glob:
        files = target_dir.glob(file_pattern)

        for fv_file in files:
            fname_with_prefix = f"{symlink_prefix}.{fv_file.name}"
            new_fv_file = timestamp_dir.joinpath(fname_with_prefix)
            new_fv_file.symlink_to(fv_file)



@pytest.mark.regression
def test_nudge_run(tmpdir_restart_dir):
    tmpdir, restart_dir = tmpdir_restart_dir
    config = get_nudging_config(default_config, restart_dir)
    fv3config.run_native(config, str("/tmp/outdir/"), capture_output=True, runfile=RUNFILE_PY)
