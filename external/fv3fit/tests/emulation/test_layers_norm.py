from logging import warning
from os import TMP_MAX
from fv3fit.emulation.layers.norm import MaxFeatureStdDenormLayer, StandardDenormLayer, StandardNormLayer
import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation import layers

_all_layers = [
    layers.StandardNormLayer,
    layers.StandardDenormLayer,
    layers.MaxFeatureStdNormLayer,
    layers.MaxFeatureStdNormLayer,
]

@pytest.fixture(params=_all_layers)
def layer_cls(request):
    return request.param


@pytest.fixture
def tensor():
    """
    Tensor with 2 features (columns)
    and 2 samples (rows)
    """
    
    return tf.Variable(
        [[0.0, 0.0],
         [1.0, 2.0]],
        dtype=tf.float32
    )




@pytest.mark.parametrize(
    "norm_cls, denorm_cls, expected",
    [
        (
            layers.StandardNormLayer,
            layers.StandardDenormLayer,
            [[-1.0, -1.0],
             [1.0, 1.0]]
        ),
        (
            layers.MaxFeatureStdNormLayer,
            layers.MaxFeatureStdDenormLayer,
            [[-0.5, -1.0],
             [0.5, 1.0]]
        )

    ]
)
def test_normalize_layers(tensor, norm_cls, denorm_cls, expected):
    norm_layer = norm_cls()
    denorm_layer = denorm_cls()
    norm_layer.fit(tensor)
    denorm_layer.fit(tensor)

    norm = norm_layer(tensor)
    expected = np.array(expected)
    np.testing.assert_allclose(norm, expected, rtol=1e-6)
    denorm = denorm_layer(norm)
    np.testing.assert_allclose(denorm, tensor, rtol=1e-6, atol=1e-6)


def test_layers_no_trainable_variables(tensor, layer_cls):
    layer = layer_cls()
    layer(tensor)

    assert [] == layer.trainable_variables


def test_standard_layers_gradient_works_epsilon(tensor):
    norm_layer = layers.StandardNormLayer()

    with tf.GradientTape(persistent=True) as tape:
        y = norm_layer(tensor)

    g = tape.gradient(y, tensor)
    expected = 1 / (norm_layer.sigma + norm_layer.epsilon)
    np.testing.assert_array_almost_equal(expected, g[0, :])


def test_warn_on_unfit_layer(tensor, layer_cls):
    layer = layer_cls()
    with pytest.warns(UserWarning):
        layer(tensor)


def test_fit_layers_are_fitted(tensor, layer_cls):
    layer = layer_cls()

    assert not layer.fitted
    layer.fit(tensor)
    assert layer.fitted
