from copy import deepcopy
from datetime import datetime, timedelta
import numpy as np
from typing import List, Mapping, Optional

import fv3config
import vcm

# this module assumes that analysis files are at 00Z, 06Z, 12Z and 18Z
SECONDS_IN_HOUR = 60 * 60
NUDGE_HOURS = np.array([0, 6, 12, 18])  # hours at which analysis data is available
NUDGE_FILE_TARGET = "INPUT"  # where to put analysis files in rundir


def _most_recent_nudge_time(start_time: datetime) -> datetime:
    """Return datetime object for the last nudging time preceding or concurrent
     with start_time"""
    first_nudge_hour = _most_recent_hour(start_time.hour)
    return datetime(start_time.year, start_time.month, start_time.day, first_nudge_hour)


def _most_recent_hour(current_hour, hour_array=NUDGE_HOURS) -> int:
    """Return latest hour in hour_array that precedes or is concurrent with
    current_hour"""
    first_nudge_hour = hour_array[np.argmax(hour_array > current_hour) - 1]
    return first_nudge_hour


def _get_nudge_time_list(config: Mapping) -> List[datetime]:
    """Return list of datetime objects corresponding to times at which analysis files
    are required for nudging for a given model run configuration"""
    current_date = config["namelist"]["coupler_nml"]["current_date"]
    start_time = datetime(*current_date)
    first_nudge_time = _most_recent_nudge_time(start_time)
    run_duration = fv3config.get_run_duration(config)
    nudge_duration = run_duration + (start_time - first_nudge_time)
    nudge_duration_hours = int(
        np.ceil(nudge_duration.total_seconds() / SECONDS_IN_HOUR)
    )
    nudge_interval = NUDGE_HOURS[1] - NUDGE_HOURS[0]
    nudging_hours = range(0, nudge_duration_hours + nudge_interval, nudge_interval)
    return [first_nudge_time + timedelta(hours=hour) for hour in nudging_hours]


def _get_nudge_filename_list(config: Mapping) -> List[str]:
    """Return list of filenames of all nudging files required"""
    nudge_filename_pattern = config["gfs_analysis_data"]["filename_pattern"]
    time_list = _get_nudge_time_list(config)
    return [time.strftime(nudge_filename_pattern) for time in time_list]


def _get_nudge_files_asset_list(config: Mapping, copy_method: str) -> List[Mapping]:
    """Return list of fv3config assets for all nudging files required for a given
    model run configuration"""
    nudge_url = config["gfs_analysis_data"]["url"]
    return [
        fv3config.get_asset_dict(
            nudge_url, file, target_location=NUDGE_FILE_TARGET, copy_method=copy_method
        )
        for file in _get_nudge_filename_list(config)
    ]


def _get_input_fname_list_asset(config: Mapping, filename: str) -> Mapping:
    fname_list_contents = "\n".join(_get_nudge_filename_list(config))
    data = fname_list_contents.encode()
    return fv3config.get_bytes_asset_dict(
        data, target_location="", target_name=filename
    )


def enable_nudge_to_observations(
    config: Mapping, file_list_path="nudging_file_list", copy_method="copy",
) -> Mapping:
    """Enable a nudged to observation run

    This sets background namelist options and adds the necessary analysis
    data files to the patch_files. To actually include nudging, the user must
    enable the nudging for each field and set the coresponding timescale.
    
    For example, this can be done using the following user configuration::
  
        namelist:
            fv_nwp_nudge_nml:
                nudge_hght: false
                nudge_ps: true
                nudge_virt: true
                nudge_winds: true
                nudge_q: true
                tau_ps: 21600.0
                tau_virt: 21600.0
                tau_winds: 21600.0
                tau_q: 21600.0


    Note:
        This function appends to patch_files and alters the namelist
    """
    config = deepcopy(config)

    default_analysis_overlay = {
        "gfs_analysis_data": {
            "url": "gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data",
            "filename_pattern": "%Y%m%d_%HZ_T85LR.nc",
        }
    }

    config = vcm.update_nested_dict(
        default_analysis_overlay,
        config,
        _namelist_overlay(input_fname_list=file_list_path),
    )

    fname_list_asset = _get_input_fname_list_asset(config, file_list_path)

    if config["gfs_analysis_data"]["url"].startswith("gs://") and copy_method == "link":
        raise ValueError(
            "Cannot link GFS analysis files if using GCS url. Use copy_method='copy'."
        )

    patch_files = config.setdefault("patch_files", [])
    patch_files.append(fname_list_asset)
    patch_files.extend(_get_nudge_files_asset_list(config, copy_method))

    return config


def _namelist_overlay(input_fname_list="nudging_file_list",) -> Mapping:
    return {
        "namelist": {
            "fv_core_nml": {"nudge": True},
            "gfs_physics_nml": {"use_analysis_sst": True},
            "fv_nwp_nudge_nml": {
                "add_bg_wind": False,
                "do_ps_bias": False,
                "ibtrack": True,
                "input_fname_list": input_fname_list,
                "k_breed": 10,
                "kbot_winds": 0,
                "mask_fac": 0.2,
                "nf_ps": 3,
                "nf_t": 3,
                "nudge_debug": True,
                "r_hi": 5.0,
                "r_lo": 3.0,
                "r_min": 225000.0,
                "t_is_tv": False,
                "tc_mask": True,
                "time_varying": False,
                "track_file_name": "No_File_specified",
                "use_high_top": True,
            },
        },
    }
