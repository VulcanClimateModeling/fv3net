import dacite
import dataclasses
from typing import Any, List, Mapping, Sequence
import tensorflow as tf

from .layers import (
    LinearBlock,
    ResidualOutput,
    FieldOutput,
    FieldInput,
    RNNBlock,
    MLPBlock,
)


def get_architecture_cls(key):

    if key == "rnn":
        return RNNBlock
    elif key == "dense":
        return MLPBlock
    elif key == "linear":
        return LinearBlock
    else:
        raise KeyError(f"Unrecognized architecture provided: {key}")


@dataclasses.dataclass
class Config:

    """
    Microphysics emulator model builder

    Args:
        input_variables: names of all inputs to the model
        output_variables: names of primary outputs of the model. Built model
            may have additional tendency output variables if specified.
        architecture: Underlying model architecture to use for the emulator.
            See `get_architecture_cls` for a list of supported layers.
        arch_params: keyword arguments to pass to the initialization
            of the architecture layer
        normalize_key: Normalization style to use for inputs/outputs.  Pass None
            to disable normalization
        selection_map: Subselection mapping for feature dimension of input/output
            variables to slices along the feature dimension
        residual_to_state: Mapping of output variable names to use a
            ResidualOutput layer for.  The value for each key should link to
            to input variable.  Analogous to learning model tendencies instead
            of direct field-to-field prediction.
        tendency_outputs: Additional outputs (tendencies) to get from the
            residual outputs.  The mapping key should match a variable in
            residual_to_state and the value is set as the output variable name
        timestep_increment_sec: Time increment multiplier for the state-tendency
            update
    """

    input_variables: List[str]
    output_variables: List[str]
    architecture: str = "rnn"
    arch_params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    normalize_key: str = "mean_std"
    selection_map: Mapping[str, slice] = dataclasses.field(default_factory=dict)
    residual_to_state: Mapping[str, str] = dataclasses.field(default_factory=dict)
    tendency_outputs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    timestep_increment_sec: int = 900

    @classmethod
    def from_dict(cls, dict_) -> "Config":
        return dacite.from_dict(cls, dict_, dacite.Config(strict=True))

    def _get_processed_inputs(self, sample_in, inputs):

        inputs = [
            FieldInput(
                sample_in=sample,
                normalize=self.normalize_key,
                selection=self.selection_map.get(name, None),
                name=f"processed_{name}",
            )(tensor)
            for name, sample, tensor in zip(self.input_variables, sample_in, inputs)
        ]

        return inputs

    def _tend_out_from_residual(self, name, residual: ResidualOutput, net_output):
        tend_name = self.tendency_outputs[name]
        tendency = residual.get_tendency_output(net_output)
        return tf.keras.layers.Lambda(lambda x: x, name=tend_name)(tendency)

    def _get_residual_output(self, name, sample, state_map, net_output):

        # incremented state field output
        res_out = ResidualOutput(
            sample,
            self.timestep_increment_sec,
            denormalize=self.normalize_key,
            name=name,
        )
        in_state = state_map[name]

        # grab tendency output if desired
        if name in self.tendency_outputs:
            tendency = [self._tend_out_from_residual(
                name, res_out, net_output
            )]
        else:
            tendency = []

        return res_out([in_state, net_output]), tendency

    def _get_outputs(self, sample_out, net_output, state_map_for_residuals):

        outputs = []
        tendencies = []
        for name, sample in zip(self.output_variables, sample_out):
            if name in state_map_for_residuals:
                out_, tendency = self._get_residual_output(name, sample, state_map_for_residuals, net_output)
                tendencies += tendency
            else:
                out_ = FieldOutput(sample, denormalize=self.normalize_key, name=name)
                out_ = out_(net_output)

            outputs.append(out_)

        return outputs + tendencies

    def build(self, sample_in: Sequence[tf.Tensor], sample_out: Sequence[tf.Tensor]):
        """
        Build model described by the configuration

        Args:
            sample_in: Sample input tensors for determining layer shapes and
                fitting normalization layers if specifies
            sample_out: Sample out tensors for determining layer shapes and
                fitting denormalization layers if specifies
        """

        inputs = [
            tf.keras.layers.Input(sample.shape[-1], name=name)
            for name, sample in zip(self.input_variables, sample_in)
        ]
        residual_map = {
            tend_name: inputs[self.input_variables.index(input_name)]
            for tend_name, input_name in self.residual_to_state.items()
        }
        processed = self._get_processed_inputs(sample_in, inputs)
        arch_layer = get_architecture_cls(self.architecture)(**self.arch_params)(
            processed
        )
        outputs = self._get_outputs(sample_out, arch_layer, residual_map)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        return model
