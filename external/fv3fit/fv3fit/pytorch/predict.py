from fv3fit._shared.predictor import Dumpable, Loadable
from .._shared.scaler import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from typing import Hashable, Iterable, Mapping, Tuple, TypeVar, Type, IO, Protocol
import zipfile
from fv3fit.pytorch.system import DEVICE
import os
import yaml
import vcm


L = TypeVar("L", bound="BinaryLoadable")


class BinaryLoadable(Protocol):
    """
    Abstract base class for objects that can be dumped.
    """

    @classmethod
    def load(cls: Type[L], f: IO[bytes]) -> L:
        """Serialize to a directory."""
        ...


def dump_mapping(mapping: Mapping[Hashable, StandardScaler], f: IO[bytes]) -> None:
    """
    Serialize a mapping to a zip file.
    """
    with zipfile.ZipFile(f, "w") as archive:
        for key, value in mapping.items():
            with archive.open(str(key), "w") as f_dump:
                value.dump(f_dump)


def load_mapping(cls: Type[L], f: IO[bytes]) -> Mapping[Hashable, L]:
    """
    Serialize a mapping to a zip file.
    """
    with zipfile.ZipFile(f, "r") as archive:
        return {name: cls.load(archive.open(name, "r")) for name in archive.namelist()}


class PytorchModel(Dumpable, Loadable):

    _MODEL_FILENAME = "weight.pt"
    _CONFIG_FILENAME = "config.yaml"
    _SCALERS_FILENAME = "scalers.zip"

    def __init__(
        self,
        state_variables: Iterable[Hashable],
        model: nn.Module,
        scalers: Mapping[Hashable, StandardScaler],
    ):
        """Initialize the predictor
        Args:
            state_variables: names of state variables
            model: pytorch model to wrap
            scalers: normalization data for each of the state variables
        """
        self.state_variables = state_variables
        self.model = model
        self.scalers = scalers

    def pack_to_tensor(self, ds: xr.Dataset, times_per_window: int) -> torch.Tensor:
        """
        Packs the dataset into a tensor to be used by the pytorch model.

        Subdivides the dataset evenly into non-overlapping windows
        of size times_per_window.

        Args:
            ds: dataset containing values to pack
            times_per_window: number of times to include per window

        Returns:
            tensor of shape [sample, time, tile, x, y, feature]
        """
        expected_dims = ("time", "tile", "x", "y", "z")
        ds = ds.transpose(*expected_dims)
        n_times = ds.time.size
        n_windows = int(n_times / times_per_window)
        # times need to be evenly divisible into windows
        ds = ds.isel(time=slice(None, n_windows * times_per_window))
        all_data = []
        for varname in self.state_variables:
            var_dims = ds[varname].dims
            if tuple(var_dims[:4]) != expected_dims[:4]:
                raise ValueError(
                    f"received variable {varname} with "
                    f"unexpected dimensions {var_dims}"
                )
            data = ds[varname].values
            data = self.scalers[varname].normalize(data)
            # segment time axis into windows
            data = data.reshape(n_windows, times_per_window, *data.shape[1:])
            if "z" not in var_dims:
                # need a z-axis for concatenation into feature axis
                data = data[..., np.newaxis]
            all_data.append(data)
        concatenated_data = np.concatenate(all_data, axis=-1)
        return torch.as_tensor(concatenated_data).float().to(DEVICE)

    def unpack_tensor(self, data: torch.Tensor) -> xr.Dataset:
        """
        Unpacks the tensor into a dataset.

        Args:
            data: tensor of shape [window, time, tile, x, y, feature]

        Returns:
            xarray dataset with values of shape [window, time, tile, x, y, feature]
        """
        i_feature = 0
        data_vars = {}
        all_dims = ["window", "time", "tile", "x", "y", "z"]
        for varname in self.state_variables:
            mean_value = self.scalers[varname].mean
            if mean_value is None:
                raise RuntimeError(f"scaler for {varname} has not been fit")
            else:
                if len(mean_value.shape) > 0 and mean_value.shape[0] > 1:
                    n_features = mean_value.shape[0]
                    var_data = data[..., i_feature : i_feature + n_features]
                else:
                    n_features = 1
                    var_data = data[..., i_feature]
                data_vars[varname] = xr.DataArray(
                    data=var_data, dims=all_dims[: len(var_data.shape)]
                )
                i_feature += n_features
        return xr.Dataset(data_vars=data_vars)

    def step_model(self, state: torch.Tensor, timesteps: int):
        """
        Step the model forward.
        Args:
            state: tensor of shape [sample, tile, x, y, feature]
            timesteps: number of timesteps to predict
        Returns:
            tensor of shape [sample, time, tile, x, y, feature], with time dimension
                having length timesteps + 1 and including the initial state
        """
        outputs = torch.zeros(
            [state.shape[0]] + [timesteps + 1] + list(state.shape[1:])
        )
        outputs[:, 0, :] = state
        for i in range(timesteps):
            with torch.no_grad():
                outputs[:, i + 1, :] = self.model(outputs[:, i, :])
        return outputs

    def predict(self, X: xr.Dataset, timesteps: int) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Predict an output xarray dataset from an input xarray dataset.

        Note that returned datasets include the initial state of the prediction,
        where by definition the model will have perfect skill.

        Args:
            X: input dataset
            timesteps: number of timesteps to predict

        Returns:
            predicted: predicted timeseries data
            reference: true timeseries data from the input dataset
        """
        tensor = self.pack_to_tensor(X, times_per_window=timesteps + 1)
        outputs = self.step_model(tensor[:, 0, :], timesteps=timesteps)
        predicted = self.unpack_tensor(outputs)
        reference = self.unpack_tensor(tensor)
        return predicted, reference

    @classmethod
    def load(cls, path: str) -> "PytorchModel":
        """Load a serialized model from a directory."""
        fs = vcm.get_fs(path)
        model_filename = os.path.join(path, cls._MODEL_FILENAME)
        with fs.open(model_filename, "rb") as f:
            model = torch.load(f)
        with fs.open(os.path.join(path, cls._SCALERS_FILENAME), "rb") as f:
            scalers = load_mapping(StandardScaler, f)
        with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        obj = cls(
            state_variables=config["state_variables"], model=model, scalers=scalers,
        )
        return obj

    def dump(self, path: str) -> None:
        fs = vcm.get_fs(path)
        model_filename = os.path.join(path, self._MODEL_FILENAME)
        with fs.open(model_filename, "wb") as f:
            torch.save(self.model, model_filename)
        with fs.open(os.path.join(path, self._SCALERS_FILENAME), "wb") as f:
            dump_mapping(self.scalers, f)
        with fs.open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
            f.write(yaml.dump({"state_variables": self.state_variables}))
