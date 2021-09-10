import dataclasses
import functools
import logging
from typing import Hashable, Mapping, MutableMapping, Set

import cftime
import fsspec
import xarray as xr

import fv3gfs.util
from runtime.monitor import Monitor
from runtime.types import Diagnostics, Step
from runtime.derived_state import DerivedFV3State

QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OverriderConfig:
    """Configuration for overriding tendencies from a step.
    
    Attributes:
        url: path to dataset containing tendencies.
        variables: mapping from state name to name of corresponding tendency in
            provided dataset. For example: {"air_temperature": "fine_res_Q1"}.
    """

    url: str
    variables: Mapping[str, str]


@dataclasses.dataclass
class OverriderAdapter:
    """Wrap a Step function to override certain tendencies."""

    config: OverriderConfig
    state: DerivedFV3State
    communicator: fv3gfs.util.CubedSphereCommunicator
    timestep: float
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self: "OverriderAdapter"):
        if self.communicator.rank == 0:
            logger.debug(f"Opening tendency overriding dataset from: {self.config.url}")
        tile = self.communicator.partitioner.tile_index(self.communicator.rank)
        mapper = fsspec.get_mapper(self.config.url)
        ds = xr.open_zarr(mapper, consolidated=True)
        ds = ds[list(self.config.variables.values())].isel(tile=tile)
        self._tendency_ds = DatasetCachedByChunk(ds, "time")

    def _open_tendencies_dataset(self, time: cftime.DatetimeJulian) -> xr.Dataset:
        ds = xr.Dataset()
        if self.communicator.tile.rank == 0:
            ds = self._tendency_ds.load(time).drop_vars("time")
        tendencies = self.communicator.tile.scatter_state(_ds_to_quantity_state(ds))
        return _quantity_state_to_ds(tendencies)

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    def override(self, name: str, func: Step) -> Diagnostics:
        if self.communicator.rank == 0:
            logger.debug(f"Overriding tendencies for {name}.")
        tendencies = self._open_tendencies_dataset(self.state.time)
        before = self.monitor.checkpoint()
        diags = func()
        change_due_to_func = self.monitor.compute_change(name, before, self.state)
        for variable_name, tendency_name in self.config.variables.items():
            with xr.set_options(keep_attrs=True):
                self.state[variable_name] = (
                    before[variable_name] + tendencies[tendency_name] * self.timestep
                )
        change_due_to_overriding = self.monitor.compute_change(
            "override", before, self.state
        )
        return {**diags, **change_due_to_func, **change_due_to_overriding}

    def __call__(self, name: str, func: Step) -> Step:
        """Override tendencies from a function that updates the State.
        
        Args:
            name: a label for the step that is being overriden.
            func: a function that updates the State and return Diagnostics.
            
        Returns:
            overridden_func: a function which observes the change to State
            done by ``func`` and overrides the change for specified variables.
        """

        def step() -> Diagnostics:
            return self.override(name, func)

        step.__name__ = func.__name__
        return step


def _ds_to_quantity_state(ds: xr.Dataset) -> QuantityState:
    quantity_state: QuantityState = {
        variable: fv3gfs.util.Quantity.from_data_array(ds[variable])
        for variable in ds.data_vars
    }
    return quantity_state


def _quantity_state_to_ds(quantity_state: QuantityState) -> xr.Dataset:
    ds = xr.Dataset(
        {
            variable: quantity_state[variable].data_array
            for variable in quantity_state.keys()
        }
    )
    return ds


class DatasetCachedByChunk:
    def __init__(self, ds: xr.Dataset, dim: str):
        self._ds = ds
        self._dim = dim
        self._chunksize = ds.chunks[dim][0]
        self._index = ds.get_index(dim)

    def load(self, key):
        idx = self._index.get_loc(key)
        chunknum, chunk_idx = divmod(idx, self._chunksize)
        return self._load_chunk(chunknum).isel({self._dim: chunk_idx})

    @functools.lru_cache(3)
    def _load_chunk(self, i):
        chunk_slice = slice(i * self._chunksize, (i + 1) * self._chunksize)
        return self._ds.isel({self._dim: chunk_slice}).load()
