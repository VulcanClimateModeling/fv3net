import os

import dataclasses

import dacite
import pytest
from runtime.segmented_run.prepare_config import (
    InitialCondition,
    HighLevelConfig,
    UserConfig,
    instantiate_dataclass_from,
)
from runtime.diagnostics.fortran import FortranFileConfig
from runtime.segmented_run import prepare_config

TEST_DATA_DIR = "tests/prepare_config_test_data"


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([f"{TEST_DATA_DIR}/prognostic_config.yml"], id="ml",),
        pytest.param([f"{TEST_DATA_DIR}/nudge_to_fine_config.yml"], id="n2f"),
        pytest.param([f"{TEST_DATA_DIR}/nudge_to_obs_config.yml"], id="n2o"),
        pytest.param([f"{TEST_DATA_DIR}/emulator.yml"], id="emulator"),
        pytest.param([f"{TEST_DATA_DIR}/fine_res_ml.yml"], id="fine-res-ml"),
    ],
)
def test_prepare_ml_config_regression(regtest, argv):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(argv)
    with regtest:
        prepare_config.prepare_config(args)


def test_get_user_config_is_valid():

    dict_ = {
        "base_version": "v0.5",
        "diagnostics": [
            {
                "name": "state_after_timestep.zarr",
                "times": {"frequency": 5400, "kind": "interval", "times": None},
                "variables": ["x_wind", "y_wind"],
            }
        ],
    }

    config = prepare_config.to_fv3config(dict_)
    # validate using dacite.from_dict
    dacite.from_dict(UserConfig, config)


def test_high_level_config_fortran_diagnostics():
    """Ensure that fortran diagnostics are translated to the Fv3config diag table"""
    config = HighLevelConfig(
        fortran_diagnostics=[FortranFileConfig(name="a", chunks={})]
    )
    dict_ = config.to_fv3config()
    # the chunk reading requires this to exist
    assert dict_["fortran_diagnostics"][0] == dataclasses.asdict(
        config.fortran_diagnostics[0]
    )
    assert len(dict_["diag_table"].file_configs) == 1


def test_instantiate_dataclass_from():
    @dataclasses.dataclass
    class A:
        a: int = 0

    @dataclasses.dataclass
    class B(A):
        b: int = 1

    b = B()
    a = instantiate_dataclass_from(A, b)
    assert a.a == b.a
    assert isinstance(a, A)


@pytest.mark.parametrize("duration, expected", [("3h", 10800), ("60s", 60)])
def test_config_high_level_duration(duration, expected):
    config = HighLevelConfig(duration=duration)
    out = config.to_fv3config()
    assert out["namelist"]["coupler_nml"]["seconds"] == expected


def test_initial_condition_default_vertical_coordinate_file():

    base_url = "/some/path"
    timestep = "20160805.000000"
    test_initial_condition = InitialCondition(base_url, timestep)

    out = test_initial_condition.vertical_coordinate_file

    assert (
        out == "gs://vcm-fv3config/data/initial_conditions/"
        "fv_core_79_levels/v1.0/fv_core.res.nc"
    )


def test_config_high_level_vertical_coordinate_file():

    base_url = "/some/path"
    timestep = "20160805.000000"
    vertical_coordinate_file = "/some/path/fv_core.res.nc"
    initial_conditions = InitialCondition(
        base_url=base_url,
        timestep=timestep,
        vertical_coordinate_file=vertical_coordinate_file,
    )

    config = HighLevelConfig(initial_conditions=initial_conditions)
    out = config.to_fv3config()

    source_location, source_name = os.path.split(vertical_coordinate_file)

    assert out["initial_conditions"][-1]["source_location"] == source_location
    assert out["initial_conditions"][-1]["source_name"] == source_name


def test_config_high_level_duration_respects_namelist():
    """The high level config should use the namelist options if the duration is
    not given"""
    config = HighLevelConfig(namelist={"coupler_nml": {"seconds": 7}})
    out = config.to_fv3config()
    assert out["namelist"]["coupler_nml"]["seconds"] == 7


def test_error_on_multiple_postphysics():

    dict_ = {
        "nudging": {"restarts_path": "/path/", "timescale_hours": {"T": 1}},
        "scikit_learn": {"model": ["/path/"]},
    }
    with pytest.raises(NotImplementedError):
        prepare_config.to_fv3config(dict_)
