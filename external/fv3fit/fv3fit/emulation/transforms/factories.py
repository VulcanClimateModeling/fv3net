import dataclasses
from typing import Callable, Optional, Sequence, Set, runtime_checkable

import tensorflow as tf
from fv3fit.emulation.transforms.transforms import (
    ComposedTransform,
    ConditionallyScaledTransform,
    TensorTransform,
    UnivariateCompatible,
    UnivariateTransform,
)
from fv3fit.emulation.types import TensorDict
from fv3fit.keras.math import groupby_bins, piecewise
from typing import Protocol


class TransformFactory(Protocol):
    """The interface of a static configuration object

    Methods:
        backward_names: used to infer to the required input variables
        build: builds the transform given some data

    """

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        pass

    def build(self, sample: TensorDict) -> TensorTransform:
        pass


@runtime_checkable
class TransformFactoryBackwardsRequiredNames(Protocol):
    """The interface of a static configuration object

    Methods:
        backward_names: used to infer to the required input variables
        backward_required_names: used to infer required input varialbes for
            backwards transformations
        build: builds the transform given some data

    """

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        pass

    def backward_input_names(self) -> Set[str]:
        pass

    def backward_output_names(self) -> Set[str]:
        pass

    def build(self, sample: TensorDict) -> TensorTransform:
        pass


@dataclasses.dataclass
class TransformedVariableConfig(TransformFactory):
    """A user facing implementation"""

    source: str
    transform: UnivariateCompatible
    to: Optional[str] = None

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        if self.to in requested_names:
            return (requested_names - {self.to}) | {self.source}
        else:
            return requested_names

    def build(self, sample: TensorDict) -> UnivariateTransform:
        return UnivariateTransform(self.source, self.transform, to=self.to)


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

        if self.to in requested_names:
            dependencies = {self.condition_on, self.source}
            requested_names = (requested_names - {self.to}) | dependencies

        return requested_names

    def backward_input_names(self) -> Set[str]:
        return {self.condition_on, self.to}

    def backward_output_names(self) -> Set[str]:
        return {self.source}

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


class ComposedTransformFactory(TransformFactoryBackwardsRequiredNames):
    def __init__(self, factories: Sequence[TransformFactory]):
        self.factories = factories

    def backward_names(self, requested_names: Set[str]) -> Set[str]:
        for factory in self.factories[::-1]:
            requested_names = factory.backward_names(requested_names)
        return requested_names

    def backward_input_names(self) -> Set[str]:
        input_names = set()
        available_vars: Set[str] = set()
        for factory in self.factories[::-1]:
            if isinstance(factory, TransformFactoryBackwardsRequiredNames):
                input_names |= factory.backward_input_names() - available_vars
                available_vars |= factory.backward_output_names()
        return input_names

    def backward_output_names(self) -> Set[str]:
        output_names = set()
        for factory in self.factories:
            if isinstance(factory, TransformFactoryBackwardsRequiredNames):
                output_names |= factory.backward_output_names()
        return output_names

    def build(self, sample: TensorDict) -> ComposedTransform:
        transforms = []
        sample = {**sample}
        for factory in self.factories:
            transform = factory.build(sample)
            sample.update(transform.forward(sample))
            transforms.append(transform)
        return ComposedTransform(transforms)
