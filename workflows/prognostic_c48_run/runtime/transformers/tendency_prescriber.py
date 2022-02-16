import dataclasses
import logging
from typing import Mapping, Set, Optional, Callable

import cftime
import xarray as xr

import pace.util
import loaders
from vcm.limit import DatasetQuantileLimiter
from runtime.monitor import Monitor
from runtime.types import Diagnostics, Step, State
from runtime.derived_state import DerivedFV3State
from runtime.conversions import quantity_state_to_dataset, dataset_to_quantity_state

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TendencyPrescriberConfig:
    """Configuration for overriding tendencies from a step.
    
    Attributes:
        mapper_config: configuration of mapper used to load tendency data.
        variables: mapping from state name to name of corresponding tendency in
            provided mapper. For example: {"air_temperature": "fine_res_Q1"}.
        reference_initial_time: if time interpolating, time of first point in dataset
        reference_frequency_seconds: time frequency of dataset
        limit_quantiles: mapping of "upper" and "lower" keys to quantile specifiers
            for limiting extremes in the Q1, Q2 dataset
    """

    mapper_config: loaders.MapperConfig
    variables: Mapping[str, str]
    reference_initial_time: Optional[str] = None
    reference_frequency_seconds: float = 900
    limit_quantiles: Optional[Mapping[str, float]] = None


@dataclasses.dataclass
class TendencyPrescriber:
    """Wrap a Step function and prescribe certain tendencies."""

    state: DerivedFV3State
    communicator: pace.util.CubedSphereCommunicator
    timestep: float
    variables: Mapping[str, str]
    mapper_func: Callable[[cftime.DatetimeJulian], State]
    limit_quantiles: Optional[Mapping[str, float]] = None
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self: "TendencyPrescriber"):
        self._limiter: Optional[DatasetQuantileLimiter] = None

    def _open_tendencies_dataset(self, time: cftime.DatetimeJulian) -> xr.Dataset:
        tile = self.communicator.partitioner.tile_index(self.communicator.rank)
        if self.communicator.tile.rank == 0:
            # https://github.com/python/mypy/issues/5485
            state = self.mapper_func(time)  # type: ignore
            ds = xr.Dataset(state).isel(tile=tile).load()
            if self._limiter is None:
                self._fit_limiter(ds)
            ds = self._limit_dataset(ds)
        else:
            ds = xr.Dataset()
        tendencies = self.communicator.tile.scatter_state(dataset_to_quantity_state(ds))
        return quantity_state_to_dataset(tendencies)

    def _fit_limiter(self, tendencies: xr.Dataset) -> None:
        if isinstance(self.limit_quantiles, dict):
            self._limiter = DatasetQuantileLimiter(
                self.limit_quantiles["upper"],
                self.limit_quantiles["lower"],
                limit_only=list(self.variables.values()),
            )
            logger.debug(f"Fitting dataset limiter with limits={self.limit_quantiles}")
            self._limiter.fit(tendencies, feature_dims=["z", "tile"])

    def _limit_dataset(self, tendencies: xr.Dataset) -> xr.Dataset:
        if self._limiter is not None:
            limited = self._limiter.transform(tendencies)
        else:
            limited = tendencies
        return limited

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    def _prescribe_tendency(self, func: Step) -> Diagnostics:
        tendencies = self._open_tendencies_dataset(self.state.time)
        before = self.monitor.checkpoint()
        diags = func()
        for variable_name, tendency_name in self.variables.items():
            with xr.set_options(keep_attrs=True):
                self.state[variable_name] = (
                    before[variable_name] + tendencies[tendency_name] * self.timestep
                )
        change_due_to_prescribing = self.monitor.compute_change(
            "tendency_prescriber", before, self.state
        )
        return {**diags, **change_due_to_prescribing}

    def __call__(self, func: Step) -> Step:
        """Override tendencies from a function that updates the State.
        
        Args:
            func: a function that updates the State and return Diagnostics.
            
        Returns:
            A function which calls ``func`` and prescribes a given change
            for specified variables.
        """

        def step() -> Diagnostics:
            return self._prescribe_tendency(func)

        step.__name__ = func.__name__
        return step
