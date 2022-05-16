from fv3fit._shared.config import RegularizerConfig
import tensorflow as tf
from typing import Optional, Sequence
import dataclasses
from .spectral_normalization import SpectralNormalization


@dataclasses.dataclass
class ConvolutionalNetwork:
    """
    Attributes:
        output: final output of convolutional network
        hidden_outputs: consecutive outputs of hidden layers in convolutional network
    """

    output: tf.Tensor
    hidden_outputs: Sequence[tf.Tensor]


class TransposeInvariant(tf.keras.constraints.Constraint):
    """Constrains `Conv2D` kernel weights to be unchanged when transposed."""

    def __call__(self, w: tf.Tensor):
        # Conv2D kernels are of shape
        # (kernel_column, kernel_row, input_channel, output_channel)
        return tf.scalar_mul(0.5, w + tf.transpose(w, perm=(1, 0, 2, 3)))


class Diffusive(tf.keras.constraints.Constraint):
    def __call__(self, w: tf.Tensor):
        w = tf.maximum(w, 0.0)
        w = tf.scalar_mul(0.5, w + tf.transpose(w, perm=(1, 0, 2, 3)))
        w = tf.scalar_mul(0.5, w + tf.experimental.numpy.fliplr(w))
        w = tf.scalar_mul(0.5, w + tf.experimental.numpy.flipud(w))
        total = tf.reduce_sum(w)
        return tf.scalar_mul(1.0 / total, w)


class ConstraintCollection(tf.keras.constraints.Constraint):
    """
    Applies given constraints sequentially.

    Note that if you give incompatible constraints, later constraints will
    take precedence and it is not guaranteed that all constraints will be
    satisfied. To know whether all constraints will be satisfied, you must
    reason about what happens when the constraints are applied sequentially.
    """

    def __init__(self, constraints: Sequence[tf.keras.constraints.Constraint]):
        super().__init__()
        self._constraints = constraints

    def __call__(self, w: tf.Tensor):
        for constraint in self._constraints:
            w = constraint(w)
        return w


@dataclasses.dataclass
class ConvolutionalNetworkConfig:
    """
    Describes how to build a convolutional network.

    The convolutional network consists of some number of convolutional layers applying
    square filters along the x- and y- dimensions (second- and third-last dimensions),
    followed by a final 1x1 convolutional layer producing the output tensor.

    In "diffusive" mode, the convolutional kernels will take a weighted mean of input
    features at nearby points to compute outputs at each step. If the number of filters
    and number of input features is 1 and 'linear' activation function is used,
    this will conserve the total amount of the input feature. Note that non-conservation
    can still occur in this mode if the output is de-scaled using different mean and
    standard deviation than the inputs.

    Attributes:
        filters: number of filters per convolutional layer, equal to
            number of neurons in each hidden layer
        depth: number of convolutional layers, including the final 1x1
            convolutional output layer. Must be greater than or equal to 1.
        kernel_size: width of convolutional filters
        kernel_regularizer: configuration of regularization for hidden layer weights
        gaussian_noise: amount of gaussian noise to add to each hidden layer output
        spectral_normalization: if True, apply spectral normalization to hidden layers
        activation_function: name of keras activation function to use on hidden layers
        transpose_invariant: if True, all layer kernels will be transpose invariant
        diffusive: if True, all layer kernels will have non-negative weights
            which sum to 1 and are equal at equal distances from the center,
            and bias will be removed from all layers
    """

    filters: int = 32
    depth: int = 3
    kernel_size: int = 3
    kernel_regularizer: RegularizerConfig = dataclasses.field(
        default_factory=lambda: RegularizerConfig("none")
    )
    gaussian_noise: float = 0.0
    spectral_normalization: bool = False
    transpose_invariant: bool = True
    diffusive: bool = False
    activation_function: str = "relu"

    def __post_init__(self):
        if self.depth < 1:
            raise ValueError("depth must be at least 1, so we can have an output layer")
        if self.filters == 0:
            raise NotImplementedError(
                "filters=0 causes a floating point exception, "
                "and we haven't written a workaround"
            )
        if self.kernel_size % 2 != 1:
            raise ValueError(
                f"kernel_size must be an odd number, got {self.kernel_size}"
            )

    @property
    def _kernel_constraint(self) -> Optional[tf.keras.constraints.Constraint]:
        constraints = []
        if self.transpose_invariant:
            constraints.append(TransposeInvariant())
        if self.diffusive:
            constraints.append(Diffusive())

        if len(constraints) == 0:
            constraint = None
        elif len(constraints) == 1:
            constraint = constraints[0]
        else:
            constraint = ConstraintCollection(constraints=constraints)
        return constraint

    @property
    def halos_required(self) -> int:
        return (self.kernel_size - 1) // 2 * (self.depth - 1)

    def build(
        self, x_in: tf.Tensor, n_features_out: int, label: str = ""
    ) -> ConvolutionalNetwork:
        """
        Take an input tensor to a convolutional network and return the result of a
        convolutional network's prediction, as tensors.

        Can be used within code that builds a larger neural network. This should
        take in and return normalized values.

        Args:
            x_in: 4+ dimensional input tensor whose last dimension is the feature
                dimension ("channel" in convolution terms), and second and third
                last dimensions are horizontal dimensions
            n_features_out: dimensionality of last (feature) dimension of output
            label: inserted into layer names, if this function is used multiple times
                to build one network you must provide a different label each time

        Returns:
            tensors resulting from the requested convolutional network, each
            tensor has the same dimensionality as the input tensor but will have fewer
            points along the x- and y-dimensions, due to convolutions
        """
        hidden_outputs = []
        x = x_in
        use_bias = not self.diffusive

        for i in range(self.depth - 1):
            if self.gaussian_noise > 0.0:
                x = tf.keras.layers.GaussianNoise(
                    self.gaussian_noise, name=f"gaussian_noise_{label}_{i}"
                )(x)
            hidden_layer = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                padding="valid",
                activation=self.activation_function,
                data_format="channels_last",
                kernel_regularizer=self.kernel_regularizer.instance,
                name=f"convolutional_{label}_{i}",
                kernel_constraint=self._kernel_constraint,
                use_bias=use_bias,
            )
            if self.spectral_normalization:
                hidden_layer = SpectralNormalization(
                    hidden_layer, name=f"spectral_norm_{label}_{i}"
                )
            x = hidden_layer(x)
            hidden_outputs.append(x)
        output = tf.keras.layers.Conv2D(
            filters=n_features_out,
            kernel_size=(1, 1),
            padding="valid",
            activation="linear",
            data_format="channels_last",
            name=f"convolutional_network_{label}_output",
            kernel_constraint=self._kernel_constraint,
            use_bias=use_bias,
        )(x)
        return ConvolutionalNetwork(hidden_outputs=hidden_outputs, output=output)
