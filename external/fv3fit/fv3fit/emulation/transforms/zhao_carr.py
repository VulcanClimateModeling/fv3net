import dataclasses
from typing import Set

import tensorflow as tf
from fv3fit.emulation.types import TensorDict
from .transforms import TensorTransform

POSITIVE_TENDENCY = "positive_tendency"
ZERO_TENDENCY = "zero_tendency"
ZERO_CLOUD = "zero_cloud"
NEGATIVE_TENDENCY = "negative_tendency"

CLASS_NAMES = {
    POSITIVE_TENDENCY,
    ZERO_TENDENCY,
    ZERO_CLOUD,
    NEGATIVE_TENDENCY,
}


@dataclasses.dataclass
class GscondClassesV1(TensorTransform):
    """
    A hardcoded classification transform to assess cloud state/tendency
    behavior
    """

    cloud_in: str = "cloud_water_mixing_ratio_input"
    cloud_out: str = "cloud_water_mixing_ratio_after_gscond"
    timestep: int = 900

    def build(self, sample: TensorDict) -> TensorTransform:
        return self

    def backward_names(self, requested_names: Set[str]) -> Set[str]:

        if CLASS_NAMES & requested_names:
            requested_names -= CLASS_NAMES
            requested_names |= {
                self.cloud_in,
                self.cloud_out,
            }
        return requested_names

    def forward(self, x: TensorDict) -> TensorDict:
        x = {**x}
        classes = classify(x[self.cloud_in], x[self.cloud_out], self.timestep)
        x.update(classes)
        return x

    def backward(self, y: TensorDict) -> TensorDict:
        return y


def classify(cloud_in, cloud_out, timestep, math=tf.math):
    state_thresh = 1e-15
    tend_thresh = 1e-15

    tend = (cloud_out - cloud_in) / timestep
    some_cloud_out = math.abs(cloud_out) > state_thresh
    negative_tend = tend < -tend_thresh

    return {
        POSITIVE_TENDENCY: tend > tend_thresh,
        ZERO_TENDENCY: math.abs(tend) <= tend_thresh,
        ZERO_CLOUD: negative_tend & ~some_cloud_out,
        NEGATIVE_TENDENCY: negative_tend & some_cloud_out,
    }