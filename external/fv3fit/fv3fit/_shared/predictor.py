import xarray as xr
import abc
from typing import Hashable, Iterable, Type, TypeVar
import logging

from .input_sensitivity import InputSensitivity

DATASET_DIM_NAME = "dataset"
logger = logging.getLogger(__file__)

L = TypeVar("L", bound="Loadable")


class Dumpable(abc.ABC):
    """
    Abstract base class for objects that can be dumped.
    """

    @abc.abstractmethod
    def dump(self, path: str) -> None:
        """Serialize to a directory."""
        pass


class Loadable(abc.ABC):
    """
    Abstract base class for objects that can be loaded from a directory.
    """

    @classmethod
    def load(cls: Type[L], path: str) -> L:
        """Load from a directory."""
        ...


class Reloadable(Dumpable, Loadable):
    """
    Abstract base class for objects that can be saved to and loaded from a directory.
    """

    pass


class Predictor(Reloadable):
    """
    Abstract base class for a predictor object, which has a `predict` method
    that takes in a stacked xarray dataset containing variables defined the class's
    `input_variables` attribute with the first dimension being the sample
    dimension, and returns predictions for the class's `output_variables` attribute.
    Also implements `load` method. Base class for model classes which implement a
    `fit` method as well, but allows creation of predictor classes to be used in
    (non-training) diagnostic and prognostic settings.
    """

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        **kwargs,
    ):
        """Initialize the predictor.

        Args:
            input_variables: names of input variables
            output_variables: names of output variables
        """
        super().__init__()
        if len(kwargs.keys()) > 0:
            raise TypeError(
                f"received unexpected keyword arguments: {tuple(kwargs.keys())}"
            )
        self.input_variables = input_variables
        self.output_variables = output_variables

    @abc.abstractmethod
    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""

    @abc.abstractmethod
    def dump(self, path: str) -> None:
        """Serialize to a directory."""
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "Predictor":
        """Load a serialized model from a directory."""
        pass

    def input_sensitivity(self, stacked_sample: xr.Dataset) -> InputSensitivity:
        """Calculate sensitivity to input features."""
        raise NotImplementedError(
            "input_sensitivity is not implemented for Predictor subclass "
            f"{self.__class__.__name__}."
        )
