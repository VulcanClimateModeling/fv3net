from runtime.steppers.machine_learning import PureMLStepper, MLStateStepper
from machine_learning_mocks import get_mock_sklearn_model
import requests
import xarray as xr
import yaml
import pytest
import vcm.testing


@pytest.fixture(scope="session")
def state(tmp_path_factory):
    url = "https://github.com/VulcanClimateModeling/vcm-ml-example-data/blob/b100177accfcdebff2546a396d2811e32c01c429/fv3net/prognostic_run/inputs_4x4.nc?raw=true"  # noqa
    r = requests.get(url)
    lpath = tmp_path_factory.getbasetemp() / "input_data.nc"
    lpath.write_bytes(r.content)
    return xr.open_dataset(str(lpath))


@pytest.fixture(params=["PureMLStepper", "MLStateStepper"])
def ml_stepper_name(request):
    return request.param


@pytest.fixture
def ml_stepper(ml_stepper_name):
    timestep = 900
    if ml_stepper_name == "PureMLStepper":
        mock_model = get_mock_sklearn_model("tendencies")
        ml_stepper = PureMLStepper(mock_model, timestep, False)
    elif ml_stepper_name == "MLStateStepper":
        mock_model = get_mock_sklearn_model("rad_fluxes")
        ml_stepper = MLStateStepper(mock_model, timestep, False)
    return ml_stepper


def test_ml_steppers_schema_unchanged(state, ml_stepper, regtest):
    (tendencies, diagnostics, states) = ml_stepper(None, state)
    xr.Dataset(diagnostics).info(regtest)
    xr.Dataset(tendencies).info(regtest)
    xr.Dataset(states).info(regtest)


def test_state_regression(state, regtest):
    checksum = vcm.testing.checksum_dataarray_mapping(state)
    print(checksum, file=regtest)


def test_ml_steppers_regression_checksum(state, ml_stepper, regtest):
    (tendencies, diagnostics, states) = ml_stepper(None, state)
    checksums = yaml.safe_dump(
        [
            ("tendencies", vcm.testing.checksum_dataarray_mapping(tendencies)),
            ("diagnostics", vcm.testing.checksum_dataarray_mapping(diagnostics)),
            ("states", vcm.testing.checksum_dataarray_mapping(states)),
        ]
    )

    print(checksums, file=regtest)
