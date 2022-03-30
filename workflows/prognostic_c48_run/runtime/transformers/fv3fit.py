import dataclasses
from typing import Mapping, Iterable, Hashable, Sequence

import xarray as xr
import fv3fit
import vcm
from runtime.steppers.machine_learning import MultiModelAdapter
from runtime.types import State
from runtime.names import SPHUM, TEMP

__all__ = ["Config", "Adapter"]


@dataclasses.dataclass
class Config:
    """
    Attributes:
        url: Sequence of paths to models that can be loaded with fv3fit.load.
        variables: Mapping from state names to name of corresponding tendency predicted
            by model. For example: {"air_temperature": "dQ1"}.
        limit_negative_humidity: if True, rescale tendencies to not allow specific
            humidity to become negative.
        online: if True, the ML predictions will be applied to model state.
    """

    url: Sequence[str]
    variables: Mapping[str, str]
    limit_negative_humidity: bool = True
    online: bool = True


@dataclasses.dataclass
class Adapter:
    config: Config
    timestep: float

    def __post_init__(self: "Adapter"):
        models = [fv3fit.load(url) for url in self.config.url]
        self.model = MultiModelAdapter(models)  # type: ignore

    def predict(self, inputs: State) -> State:
        tendencies = self.model.predict(xr.Dataset(inputs))
        if self.config.limit_negative_humidity:
            limited_tendencies = self.non_negative_sphum_limiter(tendencies, inputs)
            tendencies = tendencies.update(limited_tendencies)

        state_prediction: State = {}
        for variable_name, tendency_name in self.config.variables.items():
            with xr.set_options(keep_attrs=True):
                state_prediction[variable_name] = (
                    inputs[variable_name] + tendencies[tendency_name] * self.timestep
                )
        return state_prediction

    def apply(self, prediction: State, state: State):
        if self.config.online:
            state.update(prediction)

    def partial_fit(self, inputs: State, state: State):
        pass

    @property
    def input_variables(self) -> Iterable[Hashable]:
        return list(set(self.model.input_variables) | set(self.config.variables))

    def non_negative_sphum_limiter(self, tendencies, inputs):
        limited_tendencies = {}
        if SPHUM not in self.config.variables:
            raise NotImplementedError(
                "Cannot limit specific humidity tendencies if specific humidity "
                "updates not being predicted."
            )
        q2_name = self.config.variables[SPHUM]
        q2_new = update_q2_to_ensure_non_negative_humidity(
            inputs[SPHUM], tendencies[q2_name], self.timestep
        )
        limited_tendencies[q2_name] = q2_new
        if TEMP in self.config.variables:
            q1_name = self.config.variables[TEMP]
            q1_new = update_q1_to_conserve_mse(
                tendencies[q1_name], tendencies[q2_name], q2_new
            )
            limited_tendencies[q1_name] = q1_new
        return limited_tendencies


def update_q2_to_ensure_non_negative_humidity(
    sphum: xr.DataArray, q2: xr.DataArray, dt: float
) -> xr.DataArray:
    return xr.where(sphum + q2 * dt >= 0, q2, -sphum / dt)


def update_q1_to_conserve_mse(
    q1: xr.DataArray, q2_old: xr.DataArray, q2_new: xr.DataArray
) -> xr.DataArray:
    mse_tendency = vcm.moist_static_energy_tendency(q1, q2_old)
    q1_new = vcm.temperature_tendency(mse_tendency, q2_new)
    return q1_new
