from loaders.mappers import (
    open_fine_resolution_nudging_hybrid,
    open_fine_resolution_nudging_hybrid_clouds_off,
    open_fine_resolution_nudging_to_obs_hybrid,
)
import pytest
import synth
import numpy as np


# timestep info
timestep1 = "20160801.000730"
timestep1_end = "20160801.001500"
timestep1_npdatetime_fmt = "2016-08-01T00:15:00"
timestep2 = "20160801.002230"
timestep2_npdatetime_fmt = "2016-08-01T00:30:00"
times_numpy = [
    np.datetime64(timestep1_npdatetime_fmt),
    np.datetime64(timestep2_npdatetime_fmt),
]
times_centered_str = [timestep1, timestep2]


@pytest.fixture
def nudging_url(tmpdir):
    nudging_url = str(tmpdir.mkdir("nudging"))
    synth.generate_nudging(nudging_url, times_numpy)
    return nudging_url


@pytest.fixture
def fine_url(tmpdir):
    fine_url = str(tmpdir.mkdir("fine_res"))
    synth.generate_fine_res(fine_url)
    return fine_url, "20160801.003000"


def test_open_fine_resolution_nudging_hybrid(nudging_url, fine_url):
    # test opener
    fine_url, time_center = fine_url
    
    data = open_fine_resolution_nudging_hybrid(
        None,
        dict(url=nudging_url, nudging_variables=["air_temperature"]),
        {"fine_res_url": fine_url},
    )
    data[time_center]


@pytest.mark.parametrize(
    ("add_nudge_to_fine_tendency", "add_xshield_nudging_tendency",),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_open_fine_resolution_nudging_hybrid_clouds_off(
    nudging_url, fine_url, add_nudge_to_fine_tendency, add_xshield_nudging_tendency
):
    open_merged_nudged_kwargs = {
        "merge_files": (
            "after_physics.zarr",
            "nudging_tendencies.zarr",
            "physics_tendency_components.zarr",
        ),
        "rename_vars": {
            "air_temperature_tendency_due_to_nudging": "dQ1",
            "specific_humidity_tendency_due_to_nudging": "dQ2",
            "grid_xt": "x",
            "grid_yt": "y",
            "pfull": "z",
        },
    }
    data = open_fine_resolution_nudging_hybrid_clouds_off(
        None,
        {"url": nudging_url, "open_merged_nudged_kwargs": open_merged_nudged_kwargs},
        {"fine_res_url": fine_url},
        add_nudge_to_fine_tendency=add_nudge_to_fine_tendency,
        add_xshield_nudging_tendency=add_xshield_nudging_tendency,
    )
    data[timestep1_end]


def test_open_fine_resolution_nudging_hybrid_data_path(nudging_url, fine_url):
    # passes the urls as data_paths
    data = open_fine_resolution_nudging_hybrid([nudging_url, fine_url], {}, {})
    data[timestep1_end]


@pytest.mark.parametrize(
    "mapper_opening_function",
    [
        open_fine_resolution_nudging_hybrid,
        open_fine_resolution_nudging_hybrid_clouds_off,
    ],
)
def test_open_fine_resolution_nudging_hybrid_no_urls(
    nudging_url, fine_url, mapper_opening_function
):
    with pytest.raises(ValueError):
        mapper_opening_function(None, {}, {})
    with pytest.raises(ValueError):
        mapper_opening_function([nudging_url], {}, {})


def test_open_fine_resolution_nudging_to_obs_hybrid(nudging_url, fine_url):
    # passes the urls as mapper kwargs
    rename_prog_nudge = {
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
        "air_temperature_tendency_due_to_nudging": "t_dt_nudge",
        "specific_humidity_tendency_due_to_nudging": "q_dt_nudge",
        "grid_xt": "x",
        "grid_yt": "y",
        "pfull": "z",
    }
    nudge_kwargs = {
        "url": nudging_url,
        "merge_files": ("prognostic_diags.zarr", "nudging_tendencies.zarr"),
        "rename_vars": rename_prog_nudge,
    }
    # test opener with paths provided through kwargs
    data = open_fine_resolution_nudging_to_obs_hybrid(
        None, nudge_kwargs, {"fine_res_url": fine_url}
    )
    data[timestep1_end]


def test_open_fine_resolution_nudging_to_obs_hybrid_data_path(nudging_url, fine_url):
    # passes the urls as data_paths
    rename_prog_nudge = {
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
        "air_temperature_tendency_due_to_nudging": "t_dt_nudge",
        "specific_humidity_tendency_due_to_nudging": "q_dt_nudge",
        "grid_xt": "x",
        "grid_yt": "y",
        "pfull": "z",
    }
    nudge_kwargs = {
        "merge_files": ("prognostic_diags.zarr", "nudging_tendencies.zarr"),
        "rename_vars": rename_prog_nudge,
    }
    # test opener with paths provided through data_path
    data = open_fine_resolution_nudging_to_obs_hybrid(
        [nudging_url, fine_url], nudge_kwargs, {}
    )
    data[timestep1_end]


def test_open_fine_resolution_nudging_to_obs_hybrid_no_urls(nudging_url, fine_url):
    rename_prog_nudge = {
        "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
        "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
        "air_temperature_tendency_due_to_nudging": "t_dt_nudge",
        "specific_humidity_tendency_due_to_nudging": "q_dt_nudge",
        "grid_xt": "x",
        "grid_yt": "y",
        "pfull": "z",
    }
    nudge_kwargs = {
        "merge_files": ("prognostic_diags.zarr", "nudging_tendencies.zarr"),
        "rename_vars": rename_prog_nudge,
    }
    with pytest.raises(ValueError):
        open_fine_resolution_nudging_to_obs_hybrid(None, nudge_kwargs, {})
    with pytest.raises(ValueError):
        open_fine_resolution_nudging_to_obs_hybrid([nudging_url], nudge_kwargs, {})
