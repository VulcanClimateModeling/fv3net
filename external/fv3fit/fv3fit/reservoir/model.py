import fsspec
import numpy as np
import os
from typing import Optional, Iterable, Hashable, Union, Sequence
import xarray as xr
import yaml

import fv3fit
from fv3fit import Predictor
from .readout import ReservoirComputingReadout
from .reservoir import Reservoir
from .domain import RankDivider
from fv3fit._shared import io
from .utils import square_even_terms
from .transformers import Autoencoder, SkTransformer, ReloadableTransfomer
from ._reshaping import flatten_2d_keeping_columns_contiguous


def _load_transformer(path) -> Union[Autoencoder, SkTransformer]:
    # ensures fv3fit.load returns a ReloadableTransfomer
    model = fv3fit.load(path)

    if isinstance(model, Autoencoder) or isinstance(model, SkTransformer):
        return model
    else:
        raise ValueError(
            f"model provided at path {path} must be a ReloadableTransfomer."
        )


@io.register("hybrid-reservoir")
class HybridReservoirComputingModel(Predictor):
    _HYBRID_VARIABLES_NAME = "hybrid_variables.yaml"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        hybrid_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        reservoir: Reservoir,
        readout: ReservoirComputingReadout,
        rank_divider: RankDivider,
        square_half_hidden_state: bool = False,
        autoencoder: Optional[ReloadableTransfomer] = None,
    ):
        self.reservoir_model = ReservoirComputingModel(
            input_variables=input_variables,
            output_variables=output_variables,
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=square_half_hidden_state,
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
        self.input_variables = input_variables
        self.hybrid_variables = hybrid_variables
        self.output_variables = output_variables
        self.readout = readout
        self.square_half_hidden_state = square_half_hidden_state
        self.rank_divider = rank_divider
        self.autoencoder = autoencoder

    def predict(self, hybrid_input: np.ndarray):
        # hybrid input is assumed to be in original spatial xy dims
        # (x, y, encoded-feature) and does not include overlaps.
        # TODO: The encoding will be moved into this model

        flattened_readout_input = self._concatenate_readout_inputs(
            self.reservoir_model.reservoir.state, hybrid_input
        )
        prediction = self.readout.predict(flattened_readout_input).reshape(-1)
        return prediction

    def _concatenate_readout_inputs(self, hidden_state_input, hybrid_input):
        if self.square_half_hidden_state is True:
            hidden_state_input = square_even_terms(hidden_state_input, axis=0)
        hybrid_input = self.rank_divider.flatten_subdomains_to_columns(
            hybrid_input, with_overlap=False
        )
        readout_input = np.concatenate([hidden_state_input, hybrid_input], axis=0)
        flattened_readout_input = flatten_2d_keeping_columns_contiguous(readout_input)
        return flattened_readout_input

    def reset_state(self):
        self.reservoir_model.reset_state()

    def increment_state(self, prediction_with_overlap):
        self.reservoir_model.increment_state(prediction_with_overlap)

    def synchronize(self, synchronization_time_series):
        self.reservoir_model.synchronize(synchronization_time_series)

    def dump(self, path: str) -> None:
        self.reservoir_model.dump(path)
        with fsspec.open(os.path.join(path, self._HYBRID_VARIABLES_NAME), "w") as f:
            f.write(yaml.dump({"hybrid_variables": self.hybrid_variables}))

    @classmethod
    def load(cls, path: str) -> "HybridReservoirComputingModel":
        pure_reservoir_model = ReservoirComputingModel.load(path)
        with fsspec.open(os.path.join(path, cls._HYBRID_VARIABLES_NAME), "r") as f:
            hybrid_variables = yaml.safe_load(f)["hybrid_variables"]
        return cls(
            input_variables=pure_reservoir_model.input_variables,
            output_variables=pure_reservoir_model.output_variables,
            reservoir=pure_reservoir_model.reservoir,
            readout=pure_reservoir_model.readout,
            square_half_hidden_state=pure_reservoir_model.square_half_hidden_state,
            rank_divider=pure_reservoir_model.rank_divider,
            autoencoder=pure_reservoir_model.autoencoder,
            hybrid_variables=hybrid_variables,
        )


class HybridDatasetAdapter:
    def __init__(self, model: HybridReservoirComputingModel) -> None:
        self.model = model
        self._input_feature_sizes: Optional[Sequence] = None

    def predict(self, inputs: xr.Dataset) -> xr.Dataset:
        # TODO: centralize stacking logic for encoding decoding
        # TODO: potentially use in train.py instead of special functions there
        processed_inputs = self._input_data_to_array(inputs)  # x, y, feature dims
        prediction = self.model.predict(processed_inputs)
        unstacked_arr = self.model.rank_divider.merge_subdomains(prediction)
        return self._output_array_to_ds(unstacked_arr)

    def increment_state(self, inputs: xr.Dataset):
        processed_inputs = self._input_data_to_array(inputs)
        subdomains = self.model.rank_divider.flatten_subdomains_to_columns(
            processed_inputs, with_overlap=True
        )
        self.model.increment_state(subdomains)

    def reset_state(self):
        self.model.reset_state()

    def _encode_input_variables(self, inputs: xr.Dataset, autoencoder):
        input_arrs = [
            inputs[variable].values for variable in self.model.input_variables
        ]

        sample_dims_shape = list(input_arrs[0].shape[:-1])
        feature_len = input_arrs[0].shape[-1]
        stacked_sample_arrs = np.array(
            [arr.reshape(-1, feature_len) for arr in input_arrs]
        )
        encoded = autoencoder.encode(stacked_sample_arrs)
        encoded_shape = sample_dims_shape + [encoded.shape[-1]]
        return encoded.reshape(encoded_shape)

    def _join_input_variables(self, inputs: xr.Dataset):
        input_arrs = [inputs[variable] for variable in self.model.input_variables]
        joined_feature_inputs = np.concatenate(input_arrs, axis=-1)

        return joined_feature_inputs

    def _input_data_to_array(self, inputs: xr.Dataset):
        if self._input_feature_sizes is None:
            self._input_feature_sizes = [
                inputs[key].shape[-1] for key in self.model.input_variables
            ]

        if self.model.autoencoder is not None:
            arr = self._encode_input_variables(inputs, self.model.autoencoder)
        else:
            arr = self._join_input_variables(inputs)

        return arr

    def _output_array_to_ds(self, outputs: np.ndarray):
        if self.model.autoencoder is not None:
            var_arrays = self._decode_output_variables(outputs, self.model.autoencoder)
        else:
            var_arrays = self._separate_output_from_stacked_array(outputs)

        dims = self.model.rank_divider.rank_dims
        ds = xr.Dataset(
            {
                var: (dims, var_arrays[i])
                for i, var in enumerate(self.model.output_variables)
            }
        )

        return ds

    def _decode_output_variables(self, encoded_output: np.ndarray, autoencoder):
        if encoded_output.ndim > 3:
            raise ValueError("Unexpected dimension size in decoding operation.")

        feature_size = encoded_output.shape[-1]
        encoded_output = encoded_output.reshape(-1, feature_size)
        decoded = autoencoder.decode(encoded_output)
        spatial_shape = list(self.model.rank_divider._rank_extent_without_overlap[:-1])
        var_arrays = [arr.reshape(spatial_shape + [-1]) for arr in decoded]
        return var_arrays

    def _separate_output_from_stacked_array(self, outputs: np.ndarray):

        if self._input_feature_sizes is None:
            raise ValueError(
                "Cannot separate stacked array if input feature sizes is None."
            )

        divider = self.model.rank_divider
        split_indices = np.cumsum(self._input_feature_sizes)[:-1]
        var_arrays = np.split(outputs, split_indices, axis=-1)
        spatial_shape = list(divider._rank_extent_without_overlap[:-1])
        var_arrays = [arr.reshape(spatial_shape + [-1]) for arr in var_arrays]

        return var_arrays


@io.register("pure-reservoir")
class ReservoirComputingModel(Predictor):
    _RESERVOIR_SUBDIR = "reservoir"
    _READOUT_SUBDIR = "readout"
    _METADATA_NAME = "metadata.yaml"
    _RANK_DIVIDER_NAME = "rank_divider.yaml"
    _AUTOENCODER_SUBDIR = "autoencoder"

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        reservoir: Reservoir,
        readout: ReservoirComputingReadout,
        rank_divider: RankDivider,
        square_half_hidden_state: bool = False,
        autoencoder: Optional[ReloadableTransfomer] = None,
    ):
        """_summary_

        Args:
            reservoir: Reservoir which takes input and updates hidden state
            readout: readout layer which takes in state and predicts next time step
            square_half_hidden_state: if True, square even terms in the reservoir
                state before it is used as input to the regressor's .fit and
                .predict methods. This option was found to be important for skillful
                predictions in Wikner+2020 (https://doi.org/10.1063/5.0005541).
            rank_divider: object used to divide and reconstruct domain <-> subdomains
        """
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.reservoir = reservoir
        self.readout = readout
        self.square_half_hidden_state = square_half_hidden_state
        self.rank_divider = rank_divider
        self.autoencoder = autoencoder

    def process_state_to_readout_input(self):
        if self.square_half_hidden_state is True:
            readout_input = square_even_terms(self.reservoir.state, axis=0)
        else:
            readout_input = self.reservoir.state
        # For prediction over multiple subdomains (>1 column in reservoir state
        # array), flatten state into 1D vector before predicting
        readout_input = flatten_2d_keeping_columns_contiguous(readout_input)
        return readout_input

    def predict(self):
        # Returns raw readout prediction of latent state.
        readout_input = self.process_state_to_readout_input()
        prediction = self.readout.predict(readout_input).reshape(-1)
        return prediction

    def reset_state(self):
        input_shape = (
            self.reservoir.hyperparameters.state_size,
            self.rank_divider.n_subdomains,
        )
        self.reservoir.reset_state(input_shape)

    def increment_state(self, prediction_with_overlap):
        self.reservoir.increment_state(prediction_with_overlap)

    def synchronize(self, synchronization_time_series):
        self.reservoir.synchronize(synchronization_time_series)

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        self.reservoir.dump(os.path.join(path, self._RESERVOIR_SUBDIR))
        self.readout.dump(os.path.join(path, self._READOUT_SUBDIR))

        metadata = {
            "square_half_hidden_state": self.square_half_hidden_state,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
        }
        with fsspec.open(os.path.join(path, self._METADATA_NAME), "w") as f:
            f.write(yaml.dump(metadata))

        self.rank_divider.dump(os.path.join(path, self._RANK_DIVIDER_NAME))
        if self.autoencoder is not None:
            fv3fit.dump(self.autoencoder, os.path.join(path, self._AUTOENCODER_SUBDIR))

    @classmethod
    def load(cls, path: str) -> "ReservoirComputingModel":
        """Load a model from a remote path"""
        reservoir = Reservoir.load(os.path.join(path, cls._RESERVOIR_SUBDIR))
        readout = ReservoirComputingReadout.load(
            os.path.join(path, cls._READOUT_SUBDIR)
        )
        with fsspec.open(os.path.join(path, cls._METADATA_NAME), "r") as f:
            metadata = yaml.safe_load(f)

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]

        if fs.exists(os.path.join(path, cls._RANK_DIVIDER_NAME)):
            rank_divider = RankDivider.load(os.path.join(path, cls._RANK_DIVIDER_NAME))
        else:
            rank_divider = None

        if fs.exists(os.path.join(path, cls._AUTOENCODER_SUBDIR)):
            autoencoder: ReloadableTransfomer = _load_transformer(
                os.path.join(path, cls._AUTOENCODER_SUBDIR)
            )
        else:
            autoencoder = None  # type: ignore
        return cls(
            input_variables=metadata["input_variables"],
            output_variables=metadata["output_variables"],
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=metadata["square_half_hidden_state"],
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
