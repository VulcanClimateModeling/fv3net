"""Code for machine Learning in prognostic runs
"""
import dataclasses
import logging
import os
from typing import Hashable, Iterable, Mapping, Sequence, Set, Tuple, cast

import fv3fit
import runtime
import xarray as xr
from runtime.names import DELP, SPHUM
from runtime.types import Diagnostics, State
from runtime.diagnostics.machine_learning import rename_diagnostics
from vcm import thermo
import vcm

__all__ = ["MachineLearningConfig", "PureMLStepper", "open_model"]


logger = logging.getLogger(__name__)

NameDict = Mapping[Hashable, Hashable]


@dataclasses.dataclass
class MachineLearningConfig:
    """Machine learning configurations

    Attributes:
        model: list of URLs to fv3fit models.
        diagnostic_ml: do not apply ML tendencies if true.
        input_standard_names: mapping from non-standard names to the standard
            ones used by the model. Renames the ML inputs. Useful if the
            input variables in the ML model are inconsistent with
            the canonical names used in the wrapper.
        output_standard_names: mapping from non-standard names to the standard
            ones used by the model. Renames the ML predictions.

    Example::

        MachineLearningConfig(
            model=["gs://vcm-ml-scratch/test-annak/ml-pipeline-output"],
            diagnostic_ml=False,
            input_standard_names={},
            output_standard_names={},
        )

    """

    model: Sequence[str] = dataclasses.field(default_factory=list)
    diagnostic_ml: bool = False
    input_standard_names: Mapping[Hashable, Hashable] = dataclasses.field(
        default_factory=dict
    )
    output_standard_names: Mapping[Hashable, Hashable] = dataclasses.field(
        default_factory=dict
    )


def non_negative_sphum(
    sphum: xr.DataArray, dQ1: xr.DataArray, dQ2: xr.DataArray, dt: float
) -> Tuple[xr.DataArray, xr.DataArray]:
    delta = dQ2 * dt
    reduction_ratio = (-sphum) / (dt * dQ2)  # type: ignore
    dQ1_updated = xr.where(sphum + delta >= 0, dQ1, reduction_ratio * dQ1)
    dQ2_updated = xr.where(sphum + delta >= 0, dQ2, reduction_ratio * dQ2)
    return dQ1_updated, dQ2_updated


def _invert_dict(d: Mapping) -> Mapping:
    return dict(zip(d.values(), d.keys()))


class RenamingAdapter:
    """Adapter object for renaming model variables

    Attributes:
        model: a model to rename
        rename_in: mapping from standard names to input names of model
        rename_out: mapping from standard names to the output names of model

    """

    def __init__(
        self, model: fv3fit.Predictor, rename_in: NameDict, rename_out: NameDict = None
    ):
        self.model = model
        self.rename_in = rename_in
        self.rename_out = {} if rename_out is None else rename_out

    def _rename(self, ds: xr.Dataset, rename: NameDict) -> xr.Dataset:

        all_names = set(ds.dims) & set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        redimed = ds.rename_dims(rename_restricted)

        all_names = set(ds.data_vars) & set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        return redimed.rename(rename_restricted)

    def _rename_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        return self._rename(ds, self.rename_in)

    def _rename_outputs(self, ds: xr.Dataset) -> xr.Dataset:
        return self._rename(ds, _invert_dict(self.rename_out))

    @property
    def input_variables(self) -> Set[str]:
        invert_rename_in = _invert_dict(self.rename_in)
        return {invert_rename_in.get(var, var) for var in self.model.input_variables}

    def predict_columnwise(self, arg: xr.Dataset, **kwargs) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict_columnwise(input_, **kwargs)
        return self._rename_outputs(prediction)


class MultiModelAdapter:
    def __init__(self, models: Iterable[RenamingAdapter]):
        self.models = models

    @property
    def input_variables(self) -> Set[str]:
        vars = [model.input_variables for model in self.models]
        return {var for model_vars in vars for var in model_vars}

    def predict_columnwise(self, arg: xr.Dataset, **kwargs) -> xr.Dataset:
        predictions = []
        for model in self.models:
            predictions.append(model.predict_columnwise(arg, **kwargs))
        return xr.merge(predictions)


def open_model(config: MachineLearningConfig) -> MultiModelAdapter:
    model_paths = config.model
    models = []
    for path in model_paths:
        model = fv3fit.load(path)
        rename_in = config.input_standard_names
        rename_out = config.output_standard_names
        models.append(RenamingAdapter(model, rename_in, rename_out))
    return MultiModelAdapter(models)


def download_model(config: MachineLearningConfig, path: str) -> Sequence[str]:
    """Download models to local path and return the local paths"""
    remote_model_paths = config.model
    local_model_paths = []
    for i, remote_path in enumerate(remote_model_paths):
        local_path = os.path.join(path, str(i))
        os.makedirs(local_path)
        fs = vcm.cloud.get_fs(remote_path)
        fs.get(remote_path, local_path, recursive=True)
        local_model_paths.append(local_path)
    return local_model_paths


def predict(model: MultiModelAdapter, state: State) -> State:
    """Given ML model and state, return prediction"""
    state_loaded = {key: state[key] for key in model.input_variables}
    ds = xr.Dataset(state_loaded)  # type: ignore
    output = model.predict_columnwise(ds, feature_dim="z")
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}


class PureMLStepper:

    net_moistening = "net_moistening"

    def __init__(self, model: MultiModelAdapter, timestep: float, diagnostic_ml: bool):
        self.model = model
        self.timestep = timestep
        self.diagnostic_ml = diagnostic_ml

    def __call__(self, time, state):

        diagnostics: Diagnostics = {}
        delp = state[DELP]

        tendency: State = predict(self.model, state)

        dQ1_initial = tendency.get("dQ1", xr.zeros_like(state[SPHUM]))
        dQ2_initial = tendency.get("dQ2", xr.zeros_like(state[SPHUM]))

        dQ1_updated, dQ2_updated = non_negative_sphum(
            state[SPHUM], dQ1_initial, dQ2_initial, dt=self.timestep,
        )

        if "dQ1" in tendency:
            diag = thermo.column_integrated_heating(dQ1_updated - tendency["dQ1"], delp)
            diagnostics.update(
                {"column_integrated_dQ1_change_non_neg_sphum_constraint": (diag)}
            )
            tendency.update({"dQ1": dQ1_updated})
        if "dQ2" in tendency:
            diag = thermo.mass_integrate(dQ2_updated - tendency["dQ2"], delp, dim="z")
            diag = diag.assign_attrs({"units": "kg/m^2/s"})
            diagnostics.update(
                {"column_integrated_dQ2_change_non_neg_sphum_constraint": (diag)}
            )
            tendency.update({"dQ2": dQ2_updated})

        diagnostics["rank_updated_points"] = xr.where(dQ2_initial != dQ2_updated, 1, 0)

        state_updates = {}

        if self.diagnostic_ml:
            rename_diagnostics(diagnostics)
            tendency = {}

        return (
            tendency,
            diagnostics,
            state_updates,
        )

    def get_diagnostics(self, state, tendency):
        return runtime.compute_ml_diagnostics(state, tendency)

    def get_momentum_diagnostics(self, state, tendency):
        return runtime.compute_ml_momentum_diagnostics(state, tendency)


class MLStateStepper(PureMLStepper):
    def __call__(self, time, state):

        diagnostics: Diagnostics = {}
        state_updates: State = predict(self.model, state)

        for name in state_updates.keys():
            diagnostics[name] = state_updates[name]

        tendency = {}

        if self.diagnostic_ml:
            rename_diagnostics(diagnostics)
            state_updates = {}

        return (
            tendency,
            diagnostics,
            state_updates,
        )
