from datetime import timedelta
from cftime import DatetimeJulian as datetime
from unittest.mock import Mock

import pytest
import xarray as xr

from runtime.diagnostics import manager
from runtime.diagnostics.manager import (
    DiagnosticFile,
    DiagnosticFileConfig,
    All,
    TimeContainer,
    IntervalAveragedTimes,
)


@pytest.mark.parametrize(
    "time_stamp",
    [
        pytest.param("20160801.000000", id="full time stamp"),
        pytest.param("20160801.00000", id="time stamp with one less 0"),
    ],
)
def test_SelectedTimes(time_stamp):
    times = manager.SelectedTimes([time_stamp])
    time = datetime(year=2016, month=8, day=1, hour=0, minute=0, second=0)
    assert time in times


def test_SelectedTimes_not_in_list():
    times = manager.SelectedTimes(["20160801.000000"])
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
    times = manager.IntervalTimes(frequency, initial_time)
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
    "time, expected",
    [
        # points in interval centered at 1:30AM
        (datetime(2020, 1, 1, 0), datetime(2020, 1, 1, 1, 30)),
        (datetime(2020, 1, 1, 2, 30), datetime(2020, 1, 1, 1, 30)),
        # points in interval centered at 4:30AM
        (datetime(2020, 1, 1, 3), datetime(2020, 1, 1, 4, 30)),
    ],
)
def test_IntervalAveragedTimes_indicator(time, expected):
    times = IntervalAveragedTimes(
        frequency=timedelta(hours=3), initial_time=datetime(2000, 1, 1)
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


def test_DiagnosticFileConfig_interval_average_from_dict():
    config = DiagnosticFileConfig.from_dict(
        {
            "name": "data.zarr",
            "variables": ["a"],
            "times": {"kind": "interval-average", "frequency": 3600},
        },
        initial_time=datetime(2020, 1, 1),
    )

    assert config.times == IntervalAveragedTimes(
        timedelta(seconds=3600), datetime(2020, 1, 1)
    )
