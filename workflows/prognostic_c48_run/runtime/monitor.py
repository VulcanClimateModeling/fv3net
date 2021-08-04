import logging
from typing import (
    Callable,
    Iterable,
    Set,
)
import vcm
from runtime.types import Diagnostics, State

from .names import DELP

logger = logging.getLogger(__name__)


class Monitor:
    def __init__(self, variables: Iterable[str], state: State, timestep: float):
        # need to consume variables into set to use more than once
        var_set = set(variables)
        self.tendency_variables = filter_tendency(var_set)
        self.storage_variables = filter_storage(var_set)
        self._state = state
        self.timestep = timestep

    def __call__(
        self, name: str, func: Callable[[], Diagnostics]
    ) -> Callable[[], Diagnostics]:
        return self.monitor_tendency(name, self.monitor_storage(name, func))

    def monitor_storage(
        self, name: str, func: Callable[[], Diagnostics],
    ) -> Callable[[], Diagnostics]:
        def step() -> Diagnostics:

            vars_ = self.storage_variables
            delp_before = self._state[DELP]
            before = {key: self._state[key] for key in vars_}
            diags = func()
            delp_after = self._state[DELP]
            after = {key: self._state[key] for key in vars_}

            for variable in self.storage_variables:
                path_before = vcm.mass_integrate(before[variable], delp_before, "z")
                path_after = vcm.mass_integrate(after[variable], delp_after, "z")

                diag_name = f"storage_of_{variable}_path_due_to_{name}"
                diags[diag_name] = (path_after - path_before) / self.timestep
                if "units" in before[variable].attrs:
                    diags[diag_name].attrs["units"] = (
                        before[variable].units + " kg/m**2/s"
                    )

            mass_change = (delp_after - delp_before).sum("z") / self.timestep
            mass_change.attrs["units"] = "Pa/s"
            diags[f"storage_of_mass_due_to_{name}"] = mass_change
            return diags

        # ensure monitored function has same name as original
        step.__name__ = func.__name__
        return step

    def monitor_tendency(
        self, name: str, func: Callable[[], Diagnostics],
    ) -> Callable[[], Diagnostics]:
        def step() -> Diagnostics:

            vars_ = self.tendency_variables
            before = {key: self._state[key] for key in vars_}
            diags = func()
            after = {key: self._state[key] for key in vars_}

            # Compute statistics
            for variable in self.tendency_variables:
                diag_name = f"tendency_of_{variable}_due_to_{name}"
                diags[diag_name] = (after[variable] - before[variable]) / self.timestep
                if "units" in before[variable].attrs:
                    diags[diag_name].attrs["units"] = before[variable].units + "/s"
            return diags

        # ensure monitored function has same name as original
        step.__name__ = func.__name__
        return step


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
