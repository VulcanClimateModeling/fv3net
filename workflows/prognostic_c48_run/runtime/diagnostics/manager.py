from typing import (
    Any,
    Sequence,
    Container,
    Mapping,
    List,
    Union,
    Dict,
    Optional,
    MutableMapping,
)

import datetime
import cftime
import logging
import fv3gfs.util
import xarray as xr
import dataclasses

logger = logging.getLogger(__name__)


class All(Container):
    """A container that contains every thing

    This is useful for cases where we want an ``in`` check to always return True.

    Example:
        >>> all = All()
        >>> 'x' in all
        True
        >>> 1232.1 in all
        True
    """

    def __contains__(self, value: Any) -> bool:
        return True


class SelectedTimes(Container[cftime.DatetimeJulian]):
    TIME_FMT: str = r"%Y%m%d.%H%M%S"

    def __init__(self, times=Sequence[str]):
        self._time_stamps = times

        # see if there is an error
        self.times

    @property
    def _times(self) -> Sequence[datetime.datetime]:
        return [
            datetime.datetime.strptime(time, self.TIME_FMT)
            for time in self._time_stamps
        ]

    @property
    def times(self) -> Sequence[cftime.DatetimeJulian]:
        return [cftime.DatetimeJulian(*time.timetuple()) for time in self._times]

    def __contains__(self, time: cftime.DatetimeJulian) -> bool:
        return time in self.times


class IntervalTimes(Container[cftime.DatetimeJulian]):
    def __init__(
        self, frequency_seconds: Union[float, int], initial_time: cftime.DatetimeJulian,
    ):
        """
        Args:
            frequency_seconds: the output frequency from the initial time
            initial_time: the initial time to start the period

        """
        self._frequency_seconds = frequency_seconds
        self.initial_time = initial_time
        if self.frequency > datetime.timedelta(days=1.0) and initial_time is None:
            raise ValueError(
                "Minimum output frequency is daily when intial_time is not provided."
            )

    @property
    def frequency(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self._frequency_seconds)

    def __contains__(self, time) -> bool:
        time_since_initial_time = time - self.initial_time
        quotient = time_since_initial_time % self.frequency
        return quotient == datetime.timedelta(seconds=0)


class TimeContainer:
    """A time discretization can be described by an "indicator" function
    mapping times onto discrete set of output times.

    This generalizes the notion of a set of times to include a concept of grouping.

    """

    def __init__(self, container: Container):
        self.container = container

    def indicator(self, time: cftime.DatetimeJulian) -> Optional[cftime.DatetimeJulian]:
        """Maps a value onto set"""
        if time in self.container:
            return time
        else:
            return None


@dataclasses.dataclass
class IntervalAveragedTimes(TimeContainer):
    frequency: datetime.timedelta
    initial_time: cftime.DatetimeJulian
    includes_lower: bool = False

    def _is_endpoint(self, time: cftime.DatetimeJulian) -> bool:
        remainder = (time - self.initial_time) % self.frequency
        return remainder == datetime.timedelta(0)

    def indicator(self, time: cftime.DatetimeJulian) -> Optional[cftime.DatetimeJulian]:
        n = (time - self.initial_time) // self.frequency

        if self._is_endpoint(time) and not self.includes_lower:
            n = n - 1

        return n * self.frequency + self.frequency / 2 + self.initial_time


@dataclasses.dataclass
class TimeConfig:
    """Configuration for output times

    This class configures the time coordinate of the output diagnostics. It
    allows output data at a user-specified list of snapshots
    (``kind='selected'``), fixed intervals (``kind='interval'``), averages
    over intervals (``kind='interval-average'``), or every single time step
    (``kind='every'``).

    Attributes:
        kind: one of interval, every, "interval-average", or "selected"
        times: List of times to be used when kind=="selected". The times
            should be formatted as YYYYMMDD.HHMMSS strings. Example:
            ``["20160101.000000"]``.
        frequency: frequency in seconds, used for kind=interval-average or interval
        includes_lower: for interval-average, whether the interval includes its upper
            or lower limit. Default: False.
    """

    frequency: Optional[float] = None
    times: Optional[List[str]] = None
    kind: str = "every"
    includes_lower: bool = False

    def time_container(self, initial_time: cftime.DatetimeJulian) -> TimeContainer:
        if self.kind == "interval" and self.frequency:
            return TimeContainer(IntervalTimes(self.frequency, initial_time))
        elif self.kind == "selected":
            return TimeContainer(SelectedTimes(self.times or []))
        elif self.kind == "every":
            return TimeContainer(All())
        elif self.kind == "interval-average" and self.frequency:
            return IntervalAveragedTimes(
                datetime.timedelta(seconds=self.frequency),
                initial_time,
                self.includes_lower,
            )
        else:
            raise NotImplementedError(f"Time {self.kind} not implemented.")


@dataclasses.dataclass
class DiagnosticFileConfig:
    """Configurations for zarr Diagnostic Files

    Attributes:
        name: filename of a zarr to store the data in, e.g., 'diags.zarr'.
            Paths are relative to the run-directory root.
        variables: the variables to save. By default all available diagnostics
            are stored. Example: ``["air_temperature", "cos_zenith_angle"]``.
        remove_suffix: suffix to potentially remove from the diagnostic key
            before checking against variables and storing using the monitor.
            Applied in the flush operation.
        times: the time configuration
        chunks: mapping of dimension names to chunk sizes
    """

    name: str
    variables: Optional[Container] = None
    remove_suffix: Optional[str] = None
    times: TimeConfig = dataclasses.field(default_factory=lambda: TimeConfig())
    chunks: Optional[Mapping[str, int]] = None

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def diagnostic_file(
        self,
        initial_time: cftime.DatetimeJulian,
        partitioner: fv3gfs.util.CubedSpherePartitioner,
        comm: Any,
    ) -> "DiagnosticFile":
        return DiagnosticFile(
            variables=self.variables if self.variables else All(),
            times=self.times.time_container(initial_time),
            monitor=fv3gfs.util.ZarrMonitor(self.name, partitioner, mpi_comm=comm),
            remove_suffix=self.remove_suffix,
        )


@dataclasses.dataclass
class FortranFileConfig:
    """Configurations for Fortran diagnostics defined in diag_table to be converted to zarr

    Attributes:
        name: filename of the diagnostic. Must include .zarr suffix. For example, if
            atmos_8xdaily is defined in diag_table, use atmos_8xdaily.zarr here.
        chunks: mapping of dimension names to chunk sizes
    """

    name: str
    chunks: Mapping[str, int]

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)


class DiagnosticFile:
    """A object representing a time averaged diagnostics file

    Provides a similar interface as the "diag_table"

    Replicates the abilities of the fortran models's diag_table by allowing
    the user to specify different output times for distinct sets of
    variables.

    Note:
        Outputting a snapshot is type of time-average (e.g. taking the average
        with respect to a point mass at a given time).

    """

    def __init__(
        self,
        variables: Container,
        monitor: fv3gfs.util.ZarrMonitor,
        times: TimeContainer,
        remove_suffix: str = None,
    ):
        """
        remove_suffix: remove end of str before storing as key in Zarr

        Note:

            The containers used for times and variables do not need to be
            concrete lists or python sequences. They only need to satisfy the
            abstract ``Container`` interface. Please see the special
            containers for outputing times above:

            - ``IntervalTimes``
            - ``SelectedTimes``

            as well as the generic ``All`` container that contains the entire
            Universe!
        """
        self.variables = variables
        self.suffix = remove_suffix
        self.times = times
        self._monitor = monitor

        # variables used for averaging
        self._running_total: Dict[str, xr.DataArray] = {}
        self._current_label: Optional[cftime.DatetimeJulian] = None
        self._n = 0
        self._units: Dict[str, str] = {}

    def observe(
        self, time: cftime.DatetimeJulian, diagnostics: Mapping[str, xr.DataArray]
    ):
        for key in diagnostics:
            self._units[key] = diagnostics[key].attrs.get("units", "unknown")

        label = self.times.indicator(time)
        if label is not None:
            if label != self._current_label:
                self.flush()
                self._reset_running_average(label, diagnostics)
            else:
                self._increment_running_average(diagnostics)

    def _reset_running_average(self, label, diagnostics):
        self._running_total = {}
        for key, val in diagnostics.items():
            if key in self.variables:
                self._running_total[key] = val.copy()
        self._current_label = label
        self._n = 1

    def _increment_running_average(self, diagnostics):
        self._n += 1
        for key in diagnostics:
            if key in self.variables:
                self._running_total[key] += diagnostics[key]

    def flush(self):
        if self._current_label is not None:
            average = {key: val / self._n for key, val in self._running_total.items()}
            quantities = {
                # need units for from_data_array to work
                self._maybe_remove_suffix(key): fv3gfs.util.Quantity.from_data_array(
                    average[key].assign_attrs(units=self._units[key])
                )
                for key in average
                if key in self.variables  # isn't this always the case?
            }

            # patch this in manually. the ZarrMonitor needs it.
            # We should probably modify this behavior.
            quantities["time"] = self._current_label
            self._monitor.store(quantities)
    
    def _maybe_remove_suffix(self, key: str):
        if self.suffix is not None:
            if key.endswith(self.suffix):
                key = key[:-len(self.suffix)]

        return key

    def __del__(self):
        self.flush()


def get_diagnostic_files(
    configs: Sequence[DiagnosticFileConfig],
    partitioner: fv3gfs.util.CubedSpherePartitioner,
    comm: Any,
    initial_time: cftime.DatetimeJulian,
) -> List[DiagnosticFile]:
    """Initialize a list of diagnostic file objects from a configuration dictionary
    Note- the default here is to save all the variables in the diagnostics.
    The default set of variables can be overwritten by inserting a default diagnostics
    config entry for each runfile, e.g. ../prepare_config.py does this for
    the sklearn runfile.

    Args:
        configs: A sequence of DiagnosticFileConfigs
        paritioner: a partioner object used for writing, maybe it would be
            cleaner to pass a factory
        comm: an MPI Comm object
        initial_time: the initial time of the simulation.

    """
    return [
        config.diagnostic_file(initial_time, partitioner, comm) for config in configs
    ]


def get_chunks(
    diagnostic_file_configs: Sequence[Union[DiagnosticFileConfig, FortranFileConfig]],
) -> Mapping[str, Mapping[str, int]]:
    """Get a mapping of diagnostic file name to chunking from a sequence of diagnostic
    file configs."""
    chunks: MutableMapping = {}
    for diagnostic_file_config in diagnostic_file_configs:
        chunks[diagnostic_file_config.name] = diagnostic_file_config.chunks
    return chunks
