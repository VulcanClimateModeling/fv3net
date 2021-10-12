"""Top level factory functions

These construct objects like Emulators that require knowledge of static
configuration as well as runtime-only data structures like the model state.
"""
from typing import Optional
from runtime.types import State
from runtime.config import UserConfig
from runtime.emulator import StepTransformer, EmulatorAdapter
from runtime.tendency_prescriber import TendencyPrescriber
from runtime.derived_state import DerivedFV3State
import fv3gfs.util


__all__ = ["get_emulator_adapter", "get_tendency_prescriber"]


def get_emulator_adapter(
    config: UserConfig, state: State, timestep: float, hydrostatic: bool,
) -> Optional[StepTransformer]:
    if config.online_emulator is None:
        return None
    else:
        emulator = EmulatorAdapter(config.online_emulator)
        return StepTransformer(
            emulator,
            state,
            "emulator",
            hydrostatic,
            diagnostic_variables=set(config.diagnostic_variables),
            timestep=timestep,
        )


def get_tendency_prescriber(
    config: UserConfig,
    state: DerivedFV3State,
    timestep: float,
    communicator: fv3gfs.util.CubedSphereCommunicator,
    hydrostatic: bool,
) -> Optional[TendencyPrescriber]:
    if config.tendency_prescriber is None:
        return None
    else:
        return TendencyPrescriber(
            config.tendency_prescriber,
            state,
            communicator,
            timestep,
            hydrostatic,
            diagnostic_variables=set(config.diagnostic_variables),
        )
