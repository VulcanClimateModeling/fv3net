import fsspec
from fv3fit._shared import (
    stack_non_vertical,
    match_prediction_to_input_coords,
    stacking,
)
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Union

from fv3fit._shared.novelty_detector import NoveltyDetector
from ._shared import Predictor, io, SAMPLE_DIM_NAME
import numpy as np
import xarray as xr
import os
import yaml
from vcm import safe
import contextlib


@contextlib.contextmanager
def numpy_print_precision(precision: int):
    original_precision = np.get_printoptions()["precision"]
    original_suppress = np.get_printoptions()["suppress"]
    try:
        np.set_printoptions(precision=precision, suppress=True)
        yield
    finally:
        np.set_printoptions(precision=original_precision, suppress=original_suppress)


@io.register("constant-output")
class ConstantOutputPredictor(Predictor):
    """
    A simple predictor meant to be used for testing.

    Supports scalar and vector outputs, where the vector outputs are all
    of the same shape and assigned a dimension name of "z".
    """

    def __init__(
        self, input_variables: Iterable[Hashable], output_variables: Iterable[Hashable],
    ):
        """Initialize the predictor

        Args:
            input_variables: names of input variables
            output_variables: names of output variables

        """
        super().__init__(
            input_variables=input_variables, output_variables=output_variables,
        )
        self._outputs: Dict[Hashable, Union[np.ndarray, float]] = {}

    def set_outputs(self, **outputs: Union[np.ndarray, float]):
        """
        Set the values to output for given output variables.

        If scalars are given, the output will have dimensions [sample_dim_name],
        and if a 1D array is given, the output will have dimensions
        [sample_dim_name, z].

        Only values present in the `output_variables` attribute of this
        object will have any effect, others will be ignored silently.

        Args:
            outputs: column output for each name. For scalar
                values, use float, and for column values use 1D arrays.
        """
        self._outputs.update(outputs)  # type: ignore

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        stacked_X = stack_non_vertical(safe.get_variables(X, self.input_variables))
        n_samples = len(stacked_X[SAMPLE_DIM_NAME])
        data_vars = {}
        for name in self.output_variables:
            output = self._outputs.get(name, 0.0)
            if isinstance(output, np.ndarray):
                array = np.repeat(output[None, :], repeats=n_samples, axis=0)
                data_vars[name] = xr.DataArray(data=array, dims=[SAMPLE_DIM_NAME, "z"])
            else:
                array = np.full([n_samples], float(output))
                data_vars[name] = xr.DataArray(data=array, dims=[SAMPLE_DIM_NAME])
        coords: Optional[Mapping[Hashable, Any]] = {
            SAMPLE_DIM_NAME: stacked_X.coords[SAMPLE_DIM_NAME]
        }

        pred = xr.Dataset(data_vars=data_vars, coords=coords).unstack(SAMPLE_DIM_NAME)
        return match_prediction_to_input_coords(X, pred)

    def dump(self, path: str) -> None:
        np.savez(os.path.join(path, "_outputs.npz"), **self._outputs)
        with open(os.path.join(path, "attrs.yaml"), "w") as f:
            yaml.safe_dump(
                {
                    "input_variables": self.input_variables,
                    "output_variables": self.output_variables,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "ConstantOutputPredictor":
        """Load a serialized model from a directory."""
        outputs = dict(np.load(os.path.join(path, "_outputs.npz")))
        with open(os.path.join(path, "attrs.yaml"), "r") as f:
            attrs = yaml.safe_load(f)
        obj = cls(**attrs)
        for key, value in outputs.items():
            # loading from .npz will convert float outputs to dim-0 ndarray,
            # need to convert back to float
            if value.ndim == 0:
                outputs[key] = value.item()
        obj.set_outputs(**outputs)
        return obj


@io.register("constant-output-novelty")
class ConstantOutputNoveltyDetector(NoveltyDetector):
    """
    A simple novelty detector to be used in testing. Its score outputs are always 0
    and its novelty assessment can be changed by adjusting the score.
    """

    def __init__(self, input_variables: Iterable[Hashable]):
        super().__init__(input_variables=input_variables)

    def predict(self, data: xr.Dataset) -> xr.Dataset:
        scores_unreduced = xr.zeros_like(data[next(iter(self.input_variables))])
        unnecessary_coords = {
            k: v
            for (k, v) in scores_unreduced.coords.items()
            if k in stacking.Z_DIM_NAMES
        }
        scores_reduced = scores_unreduced.max(unnecessary_coords)
        score_dataset = scores_reduced.to_dataset(name=self._SCORE_OUTPUT_VAR)
        return score_dataset

    def _get_default_cutoff(self):
        return 0

    def dump(self, path: str) -> None:
        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]
        fs.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "attrs.yaml"), "w") as f:
            yaml.safe_dump({"input_variables": self.input_variables}, f)

    @classmethod
    def load(cls, path: str) -> "ConstantOutputNoveltyDetector":
        """Load a serialized model from a directory."""
        with open(os.path.join(path, "attrs.yaml"), "r") as f:
            attrs = yaml.safe_load(f)
        obj = cls(**attrs)
        return obj
