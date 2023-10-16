from enum import Enum
from typing import Any, Mapping
import cftime
import pace.util
import numpy as np
import pytest
import xarray as xr

from runtime.derived_state import DerivedFV3State, FV3StateMapper, MergedState


# Empty placeholder metadata for testing
TRACER_METADATA: Mapping[str, Mapping[str, Any]] = {}


class MockProperties:
    def __init__(self):
        self.DYNAMICS_PROPERTIES = []
        self.PHYSICS_PROPERTIES = [
            {
                "name": "latitude",
                "fortran_name": "xlat",
                "units": "radians",
                "container": "Grid",
                "dims": ["y", "x"],
            },
            {
                "name": "longitude",
                "fortran_name": "xlon",
                "units": "radians",
                "container": "Grid",
                "dims": ["y", "x"],
            },
        ]


class MockFV3GFS:
    def __init__(self):
        self.set_state_called = False
        self._properties = MockProperties()
        np.random.seed(0)

        nx, ny = 10, 10

        lat = pace.util.Quantity(
            np.random.rand(ny, nx), dims=["y", "x"], units="radians"
        )
        lon = pace.util.Quantity(
            np.random.rand(ny, nx), dims=["y", "x"], units="radians"
        )
        lhtfl = pace.util.Quantity(np.random.rand(ny, nx), dims=["y", "x"], units="deg")

        self.state = {
            "time": cftime.DatetimeJulian(2016, 1, 1),
            "latitude": lat,
            "longitude": lon,
            "lhtfl": lhtfl,
            "a": lhtfl,
        }

    def get_state(self, names):
        # need this for the data to remain unchanged for equality tests
        if any(name not in self.state for name in names):
            raise pace.util.InvalidQuantityError("blah")
        return {name: self.state[name] for name in names}

    def set_state_mass_conserving(self, data):

        self.set_state_called = True
        for key, value in data.items():
            assert isinstance(value, pace.util.Quantity)
            if key not in self.state:
                raise ValueError(f"{key} not in data.")
            self.state[key] = value

    def get_diagnostic_by_name(self, diagnostic):
        return self.get_state([diagnostic])[diagnostic]


class Getters(Enum):
    original = 1
    merged = 2


@pytest.fixture(params=Getters)
def getter(request):
    fv3gfs = MockFV3GFS()
    state = DerivedFV3State(fv3gfs, TRACER_METADATA)
    if request.param == Getters.original:
        return state
    elif request.param == Getters.merged:
        return MergedState(state, {})


def test_DerivedFV3State(getter):
    assert isinstance(getter["longitude"], xr.DataArray)


def test_State_keys(getter):
    assert len(getter.keys()) > 0


# test that function registered under DerivedMapping works
def test_DerivedFV3State_cos_zenith(getter):
    output = getter["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)
    assert "time" not in output.dims


def test_DerivedFV3State_latent_heat_flux(getter):
    output = getter["latent_heat_flux"]
    assert isinstance(output, xr.DataArray)


def test_DerivedFV3State_time_property(getter):
    assert isinstance(getter.time, cftime.DatetimeJulian)


def test_DerivedFV3State_time_dataarray(getter):
    assert isinstance(getter["time"], xr.DataArray)


def test_DerivedFV3State_setitem(getter):
    item = xr.DataArray([1.0], dims=["x"], attrs={"units": "m"})
    # Check that data is passed to `MockFV3GFS.set_state` correctly
    getter["a"] = item
    xr.testing.assert_equal(item, getter["a"])


def test_FV3StateMapper():
    fv3gfs = MockFV3GFS()
    mapper = FV3StateMapper(fv3gfs, TRACER_METADATA)
    assert isinstance(mapper["latitude"], xr.DataArray)


def test_FV3StateMapper_alternate_keys():
    fv3gfs = MockFV3GFS()
    mapper = FV3StateMapper(
        fv3gfs, TRACER_METADATA, alternate_keys={"lon": "longitude"}
    )
    np.testing.assert_array_almost_equal(mapper["lon"], mapper["longitude"])


def test_FV3StateMapper_raises_key_error_on_get():
    fv3gfs = MockFV3GFS()
    mapper = FV3StateMapper(fv3gfs, TRACER_METADATA)
    with pytest.raises(KeyError):
        assert "not in fv3" not in mapper
        mapper["not in fv3"]


def test_DerivedFV3State_raises_key_error_on_set():
    mapper = DerivedFV3State(MockFV3GFS(), TRACER_METADATA)
    assert "not in fv3" not in mapper
    with pytest.raises(KeyError):
        mapper["not in fv3"] = xr.DataArray(0, attrs=dict(units=""))


def _get_merged_state():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs, TRACER_METADATA)
    python_state = {}
    return getter, python_state, MergedState(getter, python_state)


def test_MergedState_time():
    wrapper, _, state = _get_merged_state()
    assert state.time == wrapper.time


def test_MergedState_use_python_state():
    wrapper, python_state, merged_state = _get_merged_state()
    item = xr.DataArray([1.0], dims=["x"], attrs={"units": "m"})
    merged_state["not in wrapper"] = item
    xr.testing.assert_equal(item, python_state["not in wrapper"])


def test_MergedState_keys():
    wrapper, _, state = _get_merged_state()
    var = "not in wrapper"
    state[var] = xr.DataArray(1.0, attrs=dict(units=""))
    assert var in state.keys()
