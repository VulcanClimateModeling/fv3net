from dataclasses import dataclass
import logging
from typing import (
    Callable,
    Iterable,
    Mapping,
    Set,
)
import xarray as xr
from runtime.types import Diagnostics, State
from runtime.diagnostics.compute import compute_change
from runtime.names import DELP

logger = logging.getLogger(__name__)

ImmutableState = Mapping[str, xr.DataArray]


@dataclass
class Monitor:
    """Utility class for monitoring changes to a state dictionary and returning
    the outputs as tendencies
    """

    tendency_variables: Set[str]
    storage_variables: Set[str]
    _state: State
    timestep: float

    def __call__(
        self, name: str, func: Callable[[], Diagnostics],
    ) -> Callable[[], Diagnostics]:
        """Decorator to add tendency monitoring to an update function

        This will add the following diagnostics:
        - `tendency_of_{variable}_due_to_{name}`
        - `storage_of_{variable}_path_due_to_{name}`. A mass-integrated version
        of the above
        - `storage_of_mass_due_to_{name}`, the total mass tendency in Pa/s.

        Args:
            name: the name to tag the tendency diagnostics with
            func: a stepping function which modifies the `state` dictionary this object
                is monitoring, but does not directly modify the `DataArray` objects
                it contains
        Returns:
            monitored function. Same as func, but with tendency and mass change
            diagnostics inserted in place
        """

        def step() -> Diagnostics:
            before = self.checkpoint()
            diags = func()
            after = self.checkpoint()
            diags.update(self.compute_change(name, before, after))
            return diags

        # ensure monitored function has same name as original
        step.__name__ = func.__name__
        return step

    @staticmethod
    def from_variables(
        variables: Iterable[str], state: State, timestep: float
    ) -> "Monitor":
        """

        Args:
            variables: list of variables with names like
                `tendency_of_{variable}_due_to_{name}`. Used to infer the variables
                to be monitored.
            state: The mutable object to monitor
            timestep: the length of the timestep used to compute the tendency
        """
        # need to consume variables into set to use more than once
        var_set = set(variables)
        return Monitor(
            tendency_variables=filter_tendency(var_set),
            storage_variables=filter_storage(var_set),
            _state=state,
            timestep=timestep,
        )

    def checkpoint(self) -> ImmutableState:
        vars_ = list(
            set(self.tendency_variables) | set(self.storage_variables) | {DELP}
        )
        before = {key: self._state[key] for key in vars_}
        return before

    def compute_change(
        self, name: str, before: ImmutableState, after: ImmutableState
    ) -> Diagnostics:
        return compute_change(
            before,
            after,
            self.tendency_variables,
            self.storage_variables,
            name,
            self.timestep,
        )


def filter_matching(variables: Iterable[str], split: str, prefix: str) -> Set[str]:
    """Get sequences of tendency and storage variables from diagnostics config."""
    return {
        variable.split(split)[0][len(prefix) :]
        for variable in variables
        if variable.startswith(prefix) and split in variable
    }


def filter_storage(variables: Iterable[str]) -> Set[str]:
    return filter_matching(variables, split="_path_due_to_", prefix="storage_of_")


def filter_tendency(variables: Iterable[str]) -> Set[str]:
    return filter_matching(variables, split="_due_to_", prefix="tendency_of_")
