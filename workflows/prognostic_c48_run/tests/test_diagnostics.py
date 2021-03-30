from datetime import timedelta
from cftime import DatetimeJulian as datetime
from unittest.mock import Mock

import fv3config
import pytest
import xarray as xr

from runtime.diagnostics.fortran import (
    FortranFileConfig,
    FortranVariableNameSpec,
    FortranTimeConfig,
    file_configs_to_namelist_settings,
)
from runtime.diagnostics.manager import DiagnosticFileConfig, DiagnosticFile, get_chunks
from runtime.diagnostics.time import (
    TimeConfig,
    All,
    TimeContainer,
    IntervalTimes,
    IntervalAveragedTimes,
    SelectedTimes,
)


@pytest.mark.parametrize(
    "time_stamp",
    [
        pytest.param("20160801.000000", id="full time stamp"),
        pytest.param("20160801.00000", id="time stamp with one less 0"),
    ],
)
def test_SelectedTimes(time_stamp):
    times = SelectedTimes([time_stamp])
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=0)
    assert time in times


def test_SelectedTimes_not_in_list():
    times = SelectedTimes(["20160801.000000"])
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=1)
    assert time not in times


august_1 = datetime(year=2016, month=8, day=1, hour=0, minute=0)
august_2 = datetime(year=2016, month=8, day=2, hour=0, minute=0)


@pytest.mark.parametrize(
    "frequency, time, initial_time, expected",
    [
        (900, datetime(year=2016, month=8, day=1, hour=0, minute=15), august_1, True),
        (900, datetime(year=2016, month=8, day=1, hour=0, minute=16), august_1, False),
        (900, datetime(year=2016, month=8, day=1, hour=12, minute=45), august_1, True),
        (86400, datetime(year=2016, month=8, day=1, hour=0, minute=0), august_1, True),
        (86400, datetime(year=2016, month=8, day=2, hour=0, minute=0), august_2, True),
        pytest.param(
            5 * 60 * 60,
            datetime(year=2016, month=8, day=2),
            datetime(year=2016, month=8, day=1),
            False,
            id="5hourlyFalse",
        ),
        pytest.param(
            5 * 60 * 60,
            datetime(year=2016, month=8, day=2, hour=1),
            datetime(year=2016, month=8, day=1),
            True,
            id="5hourlyTrue",
        ),
    ],
)
def test_IntervalTimes(frequency, time, initial_time, expected):
    times = IntervalTimes(frequency, initial_time)
    assert (time in times) == expected


def test_DiagnosticFile_time_selection():
    # mock the input data
    t1 = datetime(year=2016, month=8, day=1, hour=0, minute=15)
    t2 = datetime(year=2016, month=8, day=1, hour=0, minute=16)

    monitor = Mock()

    # observe a few times
    diag_file = DiagnosticFile(
        times=TimeContainer([t1]), variables=All(), monitor=monitor
    )
    diag_file.observe(t1, {})
    diag_file.observe(t2, {})

    # force flush to disk
    diag_file.flush()
    monitor.store.assert_called_once()


def test_DiagnosticFile_variable_selection():

    data_vars = {"a": (["x"], [1.0]), "b": (["x"], [2.0])}
    dataset = xr.Dataset(data_vars)
    diagnostics = {key: dataset[key] for key in dataset}

    class VariableCheckingMonitor:
        def store(self, state):
            assert "time" in state
            assert "a" in state
            assert "b" not in state

    monitor = VariableCheckingMonitor()

    # observe a few times
    diag_file = DiagnosticFile(
        times=TimeContainer(All()), variables=["a"], monitor=monitor
    )
    diag_file.observe(datetime(2020, 1, 1), diagnostics)
    # force flush to disk
    diag_file.flush()


@pytest.mark.parametrize(
    "attrs, expected_units", [({}, "unknown"), ({"units": "zannyunits"}, "zannyunits")]
)
def test_DiagnosticFile_variable_units(attrs, expected_units):
    data_vars = {"a": (["x"], [1.0], attrs)}
    dataset = xr.Dataset(data_vars)
    diagnostics = {key: dataset[key] for key in dataset}

    class UnitCheckingMonitor:
        def store(self, state):
            assert state["a"].units == expected_units

    monitor = UnitCheckingMonitor()

    # observe a few times
    diag_file = DiagnosticFile(
        times=TimeContainer(All()), variables=All(), monitor=monitor
    )
    diag_file.observe(datetime(2020, 1, 1), diagnostics)
    # force flush to disk
    diag_file.flush()


def test_TimeContainer_indicator():
    t = datetime(2020, 1, 1)
    time_coord = TimeContainer([t])
    assert time_coord.indicator(t) == t


def test_TimeContainer_indicator_not_present():
    t = datetime(2020, 1, 1)
    t1 = datetime(2020, 1, 1) + timedelta(minutes=1)
    time_coord = TimeContainer([t])
    assert time_coord.indicator(t1) is None


@pytest.mark.parametrize(
    "time, expected, includes_lower",
    [
        # points in interval centered at 1:30AM
        (datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 1, 30), True),
        (datetime(2020, 1, 1, 2, 30), datetime(2020, 1, 1, 1, 30), True),
        # points in interval centered at 4:30AM
        (datetime(2020, 1, 1, 3), datetime(2020, 1, 1, 4, 30), True),
        (datetime(2020, 1, 1, 3), datetime(2020, 1, 1, 1, 30), False),
        (datetime(2020, 1, 1, 2, 30), datetime(2020, 1, 1, 1, 30), False),
    ],
)
def test_IntervalAveragedTimes_indicator(time, expected, includes_lower: bool):
    times = IntervalAveragedTimes(
        frequency=timedelta(hours=3),
        initial_time=datetime(2000, 1, 1),
        includes_lower=includes_lower,
    )
    assert times.indicator(time) == expected


def test_DiagnosticFile_with_non_snapshot_time():

    t = datetime(2000, 1, 1)
    one = {"a": xr.DataArray(1.0), "b": xr.DataArray(1.0)}
    two = {"a": xr.DataArray(2.0), "b": xr.DataArray(2.0)}

    class Hours(TimeContainer):
        def __init__(self):
            pass

        def indicator(self, time):
            return t + timedelta(hours=time.hour)

    class MockMonitor:
        data = {}

        def store(self, x):
            assert isinstance(x["time"], datetime), x
            self.data[x["time"]] = x

    monitor = MockMonitor()
    diag_file = DiagnosticFile(times=Hours(), variables=["a", "b"], monitor=monitor)

    for time, x in [
        (t, one),
        (t + timedelta(minutes=30), one),
        (t + timedelta(minutes=45), one),
        (t + timedelta(hours=1, minutes=25), one),
        (t + timedelta(hours=1, minutes=35), two),
    ]:
        diag_file.observe(time, x)

    diag_file.flush()

    # there should be only two time intervals
    assert len(monitor.data) == 2

    assert monitor.data[datetime(2000, 1, 1, 0)]["a"].data.item() == pytest.approx(1.0)
    assert monitor.data[datetime(2000, 1, 1, 1)]["a"].data.item() == pytest.approx(1.5)
    assert monitor.data[datetime(2000, 1, 1, 0)]["b"].data.item() == pytest.approx(1.0)
    assert monitor.data[datetime(2000, 1, 1, 1)]["b"].data.item() == pytest.approx(1.5)


def test_TimeConfig_interval_average():
    config = TimeConfig(frequency=3600, kind="interval-average")
    container = config.time_container(datetime(2020, 1, 1))
    assert container == IntervalAveragedTimes(
        timedelta(seconds=3600), datetime(2020, 1, 1), includes_lower=False
    )


def test_TimeConfig_interval_average_endpoint():
    config = TimeConfig(frequency=3600, kind="interval-average", includes_lower=True)
    container = config.time_container(datetime(2020, 1, 1))
    assert container == IntervalAveragedTimes(
        timedelta(seconds=3600), datetime(2020, 1, 1), includes_lower=True
    )


@pytest.mark.parametrize(
    "fortran_diagnostics,diagnostics,expected_chunks",
    [
        ([], [], {}),
        (
            [FortranFileConfig(name="sfc_dt_atmos.zarr", chunks={"time": 2})],
            [],
            {"sfc_dt_atmos.zarr": {"time": 2}},
        ),
        (
            [],
            [DiagnosticFileConfig(name="diags.zarr", chunks={"time": 4})],
            {"diags.zarr": {"time": 4}},
        ),
        (
            [FortranFileConfig(name="sfc_dt_atmos.zarr", chunks={"time": 2})],
            [DiagnosticFileConfig(name="diags.zarr", chunks={"time": 4})],
            {"diags.zarr": {"time": 4}, "sfc_dt_atmos.zarr": {"time": 2}},
        ),
    ],
)
def test_get_chunks(fortran_diagnostics, diagnostics, expected_chunks):
    chunks = get_chunks(fortran_diagnostics + diagnostics)
    assert chunks == expected_chunks


@pytest.mark.parametrize(
    "time_config, expected_frequency, expected_reduction_method",
    [
        (FortranTimeConfig(kind="every"), 0, "none"),
        (FortranTimeConfig(kind="interval", frequency=900), 900, "none"),
        (FortranTimeConfig(kind="interval-average", frequency=900), 900, "average"),
    ],
)
def test_to_fv3config_diag_file_config(
    time_config, expected_frequency, expected_reduction_method
):
    fortran_file_config = FortranFileConfig(
        "atmos_8xdaily.zarr",
        {"time": 2},
        variables=[FortranVariableNameSpec("dynamics", "ta", "air_temperature")],
        times=time_config,
    )
    field = fv3config.DiagFieldConfig(
        "dynamics", "ta", "air_temperature", reduction_method=expected_reduction_method,
    )
    expected_config = fv3config.DiagFileConfig(
        "atmos_8xdaily", expected_frequency, "seconds", [field],
    )
    assert fortran_file_config.to_fv3config_diag_file_config() == expected_config


@pytest.mark.parametrize(
    "time_config, expected_frequency, raises_error",
    [
        (FortranTimeConfig(kind="every"), timedelta(seconds=0), False),
        (
            FortranTimeConfig(kind="interval", frequency=900),
            timedelta(seconds=900),
            False,
        ),
        (FortranTimeConfig(kind="interval", frequency=7200), timedelta(hours=2), False),
        (FortranTimeConfig(kind="selected"), 0, True),
    ],
)
def test_fortran_time_config_to_frequency(
    time_config, expected_frequency, raises_error
):
    if raises_error:
        with pytest.raises(NotImplementedError):
            time_config.to_frequency()
    else:
        assert time_config.to_frequency() == expected_frequency


def test_file_configs_to_namelist_settings_raises_error():
    diagnostics = [
        FortranFileConfig(
            "physics_3hourly.zarr",
            {"time": 1},
            [FortranVariableNameSpec("gfs_phys", "totprcpb_ave", "PRATEsfc")],
            FortranTimeConfig("interval", 10800),
        ),
        FortranFileConfig(
            "physics_hourly.zarr",
            {"time": 1},
            [FortranVariableNameSpec("gfs_sfc", "t2m", "TMP2m")],
            FortranTimeConfig("interval", 3600),
        ),
    ]
    with pytest.raises(NotImplementedError):
        file_configs_to_namelist_settings(diagnostics, 900)


@pytest.mark.parametrize(
    "time_config, expected_frequency_in_hours",
    [
        (FortranTimeConfig(kind="every"), 0.25),
        (FortranTimeConfig(kind="interval", frequency=900), 0.25),
        (FortranTimeConfig(kind="interval", frequency=3600), 1),
    ],
)
def test_file_configs_to_namelist_settings(time_config, expected_frequency_in_hours):
    diagnostics = [
        FortranFileConfig(
            "physics_3hourly.zarr",
            {"time": 1},
            [
                FortranVariableNameSpec("gfs_phys", "totprcpb_ave", "PRATEsfc"),
                FortranVariableNameSpec("gfs_sfc", "t2m", "TMP2m"),
            ],
            time_config,
        ),
    ]
    overlay = file_configs_to_namelist_settings(diagnostics, timedelta(seconds=900))
    expected_overlay = {
        "namelist": {
            "atmos_model_nml": {"fhout": expected_frequency_in_hours},
            "gfs_physics_nml": {"fhzero": expected_frequency_in_hours},
        }
    }
    assert overlay == expected_overlay


def test_file_configs_to_namelist_settings_empty_diagnostics():
    assert file_configs_to_namelist_settings([], timedelta(seconds=900)) == {}
