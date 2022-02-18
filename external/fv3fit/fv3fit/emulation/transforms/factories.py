import dataclasses
from typing import Callable, Optional, Sequence, Set

import tensorflow as tf
from fv3fit.emulation.transforms.transforms import (
    ComposedTransform,
    ConditionallyScaledTransform,
    LogTransform,
    TensorTransform,
    UnivariateTransform,
)
from fv3fit.emulation.types import TensorDict
from fv3fit.keras.math import groupby_bins, piecewise
from typing_extensions import Protocol


class TransformFactory(Protocol):
    """The interface of a static configuration object

    Attrs:
        to: Name of transform result

    Methods:
        backward_names: used to infer to the required input variables
        build: builds the transform given some data
        required_names: required inputs for successful transform

    """

    to: str

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        pass

    @property
    def required_names(self) -> Set[str]:
        pass

    def build(self, sample: TensorDict) -> TensorTransform:
        pass


@dataclasses.dataclass
class TransformedVariableConfig(TransformFactory):
    """A user facing implementation"""

    source: str
    to: str
    transform: LogTransform

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        if self.to in requested_names:
            return (requested_names - {self.to}) | {self.source}
        else:
            return requested_names

    def build(self, sample: TensorDict) -> TensorTransform:
        return UnivariateTransform(self.source, self.to, self.transform)

    @property
    def required_names(self) -> Set[str]:
        return {self.source}


def reduce_std(x: tf.Tensor) -> tf.Tensor:
    mean = tf.reduce_mean(x)
    return tf.sqrt(tf.reduce_mean((x - mean) ** 2))


def fit_conditional(
    x: tf.Tensor, y: tf.Tensor, reduction: Callable[[tf.Tensor], tf.Tensor], bins: int,
) -> Callable[[tf.Tensor], tf.Tensor]:
    min = tf.reduce_min(x)
    max = tf.reduce_max(x)
    edges = tf.linspace(min, max, bins + 1)
    values = groupby_bins(edges, x, y, reduction)

    def interp(x: tf.Tensor) -> tf.Tensor:
        return piecewise(edges[:-1], values, x)

    return interp


@dataclasses.dataclass
class ConditionallyScaled(TransformFactory):
    """Conditionally scaled transformation

    Scales ``source`` by conditional standard deviation and mean::

                  source - E[source|on]
        to =  --------------------------------
               max[Std[source|on], min_scale]

    Attributes:
        to: name of the transformed variable.
        condition_on: the variable to condition on
        bins: the number of bins
        source: The variable to be normalized.
        min_scale: the minimium scale to normalize by. Used when the scale might
            be 0.
        fit_filter_magnitude: if provided, any values with
            |source| < filter_magnitude are removed from the standard
            deviation/mean calculation.

    """

    to: str
    condition_on: str
    source: str
    bins: int
    min_scale: float = 0.0
    fit_filter_magnitude: Optional[float] = None

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        """List the names needed to compute ``self.to``"""
        new_names = (
            {self.condition_on, self.source} if self.to in requested_names else set()
        )
        return requested_names.union(new_names)

    @property
    def required_names(self) -> Set[str]:
        return {self.condition_on, self.source}

    def build(self, sample: TensorDict) -> ConditionallyScaledTransform:

        if self.fit_filter_magnitude is not None:
            mask = tf.abs(sample[self.source]) > self.fit_filter_magnitude
        else:
            mask = ...

        return ConditionallyScaledTransform(
            to=self.to,
            on=self.condition_on,
            source=self.source,
            scale=fit_conditional(
                sample[self.condition_on][mask],
                sample[self.source][mask],
                reduce_std,
                self.bins,
            ),
            center=fit_conditional(
                sample[self.condition_on][mask],
                sample[self.source][mask],
                tf.reduce_mean,
                self.bins,
            ),
            min_scale=self.min_scale,
        )


class ComposedTransformFactory(TransformFactory):
    def __init__(self, factories: Sequence[TransformFactory]):
        self.factories = factories

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        for factory in self.factories:
            requested_names = factory.backward_names(requested_names)
        return requested_names

    def build(self, sample: TensorDict) -> ComposedTransform:
        transforms = [factory.build(sample) for factory in self.factories]
        return ComposedTransform(transforms)


def _get_required_inputs(
    f: ComposedTransformFactory, requested_names: Set[str]
) -> Set[str]:

    for t in f.factories:

        available_from_transform: Set[str] = set()
        required: Set[str] = set()

        required |= t.required_names - available_from_transform
        available_from_transform |= {t.to}

        requested_names -= available_from_transform

    return set()


def _get_dependencies(name: str, factories: Sequence[TransformFactory]):

    deps = set()
    for i, factory in enumerate(factories[::-1], 1):

        if factory.to == name:
            for dep_name in sorted(factory.required_names):
                deps |= _get_dependencies(dep_name, factories[:-i])
            break
    else:
        return {name}

    return deps
