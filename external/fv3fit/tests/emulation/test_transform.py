from unittest.mock import Mock
import numpy as np
import pytest
import tensorflow as tf
from fv3fit.emulation.transforms import (
    ComposedTransformFactory,
    ComposedTransform,
    Difference,
    LogTransform,
    TransformedVariableConfig,
    LimitValueTransform,
    TendencyToFlux,
)
from fv3fit.emulation.transforms.transforms import ConditionallyScaledTransform
from fv3fit.emulation.transforms.factories import ConditionallyScaled, fit_conditional
from fv3fit.emulation.transforms import factories


def _to_float(x: tf.Tensor) -> float:
    return x.numpy().item()


def _assert_scalar_approx(expected, actual):
    assert _to_float(expected) == pytest.approx(_to_float(actual))


def test_LogTransform_forward_backward():
    transform = LogTransform()
    x = tf.constant(0.001)
    x_approx = transform.backward(transform.forward(x))
    _assert_scalar_approx(x, x_approx)


def _get_per_variable_mocks():
    class MockTransform:
        def forward(self, x):
            return x + 1

        def backward(self, y):
            return y - 1

    x = {"a": tf.constant(1.0), "b": tf.constant(1.0)}
    mock_xform = MockTransform()
    factory = ComposedTransformFactory(
        [TransformedVariableConfig("a", mock_xform, to="transformed")]
    )
    transform = factory.build(x)
    expected_forward = {
        "a": tf.constant(1.0),
        "b": tf.constant(1.0),
        "transformed": tf.constant(mock_xform.forward(x["a"])),
    }

    return transform, x, expected_forward


def test_per_variable_transform_forward():
    transform, x, expected_forward = _get_per_variable_mocks()
    y = transform.forward(x)
    assert set(y) == set(expected_forward)
    for key in y:
        _assert_scalar_approx(expected_forward[key], y[key])


def test_per_variable_transform_round_trip():
    transform, x, _ = _get_per_variable_mocks()
    y = transform.backward(transform.forward(x))
    assert set(y) >= set(x)
    for key in x:
        _assert_scalar_approx(x[key], y[key])


def test_variable_transform_default_to():

    transform = TransformedVariableConfig("a", LogTransform())
    built = transform.build({})

    assert built.to == "a"
    assert transform.backward_names({"a"}) == {"a"}


def test_per_variable_transform_backward_names():
    transform = ComposedTransformFactory(
        [TransformedVariableConfig("a", LogTransform(), to="b")]
    )
    assert transform.backward_names({"b"}) == {"a"}
    assert transform.backward_names({"b", "random"}) == {"a", "random"}


def test_composed_transform_backward_names_sequence():
    """intermediate names produced by one transform should not be listed in the
    required_names
    """
    transform = ComposedTransformFactory(
        [
            TransformedVariableConfig("a", LogTransform(), to="b"),
            TransformedVariableConfig("b", LogTransform(), to="c"),
        ]
    )
    assert transform.backward_names({"c"}) == {"a"}


def test_composed_transform_with_circular_dep():
    factory = ComposedTransformFactory(
        [
            TransformedVariableConfig("a", LogTransform(), to="b"),
            TransformedVariableConfig("b", LogTransform(), to="a"),
        ]
    )

    assert factory.backward_names({"a"}) == {"a"}


def test_composed_transform_ok_with_repeated_dep():
    factory = ComposedTransformFactory(
        [
            TransformedVariableConfig("a", LogTransform(), to="b"),
            TransformedVariableConfig("b", LogTransform(), to="c"),
            TransformedVariableConfig("b", LogTransform(), to="d"),
        ]
    )
    return factory.backward_names({"c", "d"}) == {"a"}


def test_ComposedTransform_forward_backward_on_sequential_transforms():
    # some transforms could be mutually dependent

    class Rename:
        def __init__(self, in_name, out_name):
            self.in_name = in_name
            self.out_name = out_name

        def forward(self, x):
            return {self.out_name: x[self.in_name]}

        def backward(self, y):
            return {self.in_name: y[self.out_name]}

    transform = ComposedTransform([Rename("a", "b"), Rename("b", "c")])
    data = {"a": tf.ones((1,))}
    assert set(transform.forward(data)) == {"c"}
    assert set(transform.backward(transform.forward(data))) == set(data)


def test_fit_conditional():
    x = tf.convert_to_tensor([0.0, 1, 2])
    y = tf.convert_to_tensor([1.0, 2, 2])
    interp = fit_conditional(x, y, tf.reduce_mean, 2)
    # bins will be 0, 1, 2
    # mean is [1, 2]
    out = interp([-1, 0.0, 1, 3])
    expected = [1, 1, 2, 2]
    np.testing.assert_array_almost_equal(expected, out)


def test_fit_conditional_2d():
    # should work with multiple dimensions
    x = tf.convert_to_tensor([0.0, 1, 2])
    y = tf.convert_to_tensor([1.0, 2, 2])
    interp = fit_conditional(x, y, tf.reduce_mean, 2)

    in_ = tf.ones((10, 10))
    out = interp(in_)
    assert out.shape == in_.shape


def _get_mocked_transform(scale_value=2.0, min_scale=0.0):
    scale = Mock()
    scale.return_value = scale_value
    center = Mock()
    center.return_value = 0.0

    input_name = "x_in"
    source_name = "x_out"
    to = "difference"

    zero = tf.zeros((3, 4), dtype=tf.float32)

    expected = 1.0

    data = {
        input_name: zero,
        source_name: zero + expected * max(scale.return_value, min_scale),
    }

    transform = ConditionallyScaledTransform(
        source=source_name,
        to=to,
        on=input_name,
        scale=scale,
        center=center,
        min_scale=min_scale,
    )

    return scale, center, transform, data, expected, to


@pytest.mark.parametrize("min_scale", [0, 1, 2, 3])
def test_ConditionallyScaledTransform_forward(min_scale: float):
    scale, center, transform, data, expected, to = _get_mocked_transform(
        min_scale=min_scale
    )
    out = transform.forward(data)
    np.testing.assert_array_almost_equal(out[to], expected)
    scale.assert_called_once()
    center.assert_called_once()


@pytest.mark.parametrize("min_scale", [0, 4])
def test_ConditionallyScaledTransform_backward(min_scale: float):
    # need a scale-value other than one to find round-tripped errors
    scale, center, transform, data, expected, to = _get_mocked_transform(
        scale_value=2.0, min_scale=min_scale
    )
    out = transform.forward(data)
    round_tripped = transform.backward(out)
    assert set(round_tripped) == set(data) | set(out)
    for key in data:
        np.testing.assert_array_almost_equal(data[key], round_tripped[key])


def test_ConditionallyScaled_backward_names():
    factory = ConditionallyScaled(source="in", to="z", bins=10, condition_on="T")
    assert factory.backward_names({"z"}) == {"T", "in"}


def test_ConditionallyScaled_backward_names_output_not_in_request():
    factory = ConditionallyScaled(source="in", to="z", bins=10, condition_on="T")
    assert factory.backward_names({"a", "b"}) == {"a", "b"}


def test_ConditionallyScaled_build():
    tf.random.set_seed(0)
    out_name = "x_out"
    on = "T"
    to = "diff"
    factory = ConditionallyScaled(source=out_name, to=to, bins=3, condition_on=on)
    shape = (10, 4)
    data = {
        out_name: tf.random.uniform(shape) * 3,
        on: tf.random.uniform(shape),
    }
    transform = factory.build(data)
    out = transform.forward(data)

    assert tf.reduce_mean(out[to]).numpy() == pytest.approx(0.0, abs=0.1)
    assert tf.reduce_mean(out[to] ** 2).numpy() == pytest.approx(1.0, abs=0.1)


def test_Difference_backward_names():
    diff = Difference("diff", "before", "after")
    assert diff.backward_names({"diff"}) == {"before", "after"}
    assert diff.backward_names({"not in a"}) == {"not in a"}


def test_Difference_build():
    diff = Difference("diff", "before", "after")
    assert diff.build({}) == diff


def test_Difference_forward():
    diff = Difference("diff", "before", "after")
    in_ = {"after": 1, "before": 0}
    assert diff.forward(in_) == {"diff": 1, **in_}


def test_Difference_backward():
    diff = Difference("diff", "before", "after")
    in_ = {"diff": 1, "before": 0}
    assert diff.backward(in_) == {"after": 1, **in_}

    # test if after is already present
    in_ = {"diff": 1, "before": 0, "after": 1000}
    assert diff.backward(in_) == {"after": 1, "before": 0, "diff": 1}


@pytest.mark.parametrize("filter_magnitude", [1e-5, 2e-5, None])
def test_ConditionallyScaled_applies_mask(monkeypatch, filter_magnitude):
    source = "x_out"
    on = "T"
    to = "x_scaled"
    shape = (1, 1)

    magnitude = 1e-5
    data = {
        source: tf.fill(shape, magnitude),
        on: tf.random.uniform(shape),
    }

    if filter_magnitude is None:
        expected_shape = shape
    elif filter_magnitude >= magnitude:
        expected_shape = (0,)
    else:
        expected_shape = shape

    # .build calls fit_conditional, so let's mock it
    fit_conditional = Mock()
    fit_conditional.return_value = lambda x: 1.0
    monkeypatch.setattr(factories, "fit_conditional", fit_conditional)

    factory = ConditionallyScaled(
        source=source,
        to=to,
        bins=1,
        condition_on=on,
        fit_filter_magnitude=filter_magnitude,
    )

    factory.build(data)

    # assert that fit_conditional was passed arrays of the expected size
    fit_conditional_x_arg = fit_conditional.call_args[0][0]
    assert fit_conditional_x_arg.shape == expected_shape


def test_ComposedTransform_with_build():
    """Check that composed transform works if an earlier transform produces an
    output need by the .build of a later one"""

    class MockTransform:
        def forward(self, x):
            return {"b": x["a"]}

    factory1 = Mock()
    factory1.build.return_value = MockTransform()

    factory2 = Mock()
    factory2.build.return_value = MockTransform()

    data = {"a": 0}

    ComposedTransformFactory([factory1, factory2]).build(data)

    # mock2.build is called with the "b" variable outputted by mock1
    (build_sample_for_second_mock,) = factory2.build.call_args[0]
    assert build_sample_for_second_mock == {"b": 0, "a": 0}


@pytest.mark.parametrize(
    "lower,upper, expected",
    [
        (None, None, [-2, -1, 0, 1, 2, 3]),
        (0, None, [0, 0, 0, 1, 2, 3]),
        (None, 0, [-2, -1, 0, 0, 0, 0]),
        (-2, 2, [0, -1, 0, 1, 0, 0]),
        (1, 1, [0, 0, 0, 0, 0, 0]),
    ],
    ids=["no limits", "lower", "upper", "lower + upper", "equivalent lower + upper"],
)
def test_PositiveTransform(lower, upper, expected):

    tensor = tf.convert_to_tensor([-2, -1, 0, 1, 2, 3])
    transform = LimitValueTransform(lower=lower, upper=upper)

    forward_result = transform.forward(tensor)
    np.testing.assert_array_equal(tensor, forward_result)

    positive = tf.convert_to_tensor(expected)
    backward_result = transform.backward(tensor)
    np.testing.assert_array_equal(positive, backward_result)


def _get_vertical_flux_transform():
    delp = tf.convert_to_tensor([1.0, 1, 2])
    interface_flux = tf.convert_to_tensor([0.5, 1, 2])
    down_sfc_flux = tf.convert_to_tensor([5.0])
    up_sfc_flux = tf.convert_to_tensor([1.5])
    x = {
        "delp": delp,
        "flux": interface_flux,
        "sfc_down": down_sfc_flux,
        "sfc_up": up_sfc_flux,
        "toa_net": interface_flux[0],
    }
    transform = TendencyToFlux(
        "tendency",
        "flux",
        "sfc_down",
        "sfc_up",
        "delp",
        net_toa_flux="toa_net",
        gravity=1.0,
    )
    expected_tendency = tf.convert_to_tensor([-0.5, -1, -0.75])
    return x, expected_tendency, transform


def test_TendencyToFlux_backward():
    x, expected_tendency, transform = _get_vertical_flux_transform()
    y = transform.backward(x)
    tf.debugging.assert_equal(y["tendency"], expected_tendency)
    for name in x:
        tf.debugging.assert_equal(x[name], y[name])


def test_TendencyToFlux_round_trip():
    x, _, transform = _get_vertical_flux_transform()
    y = transform.backward(x)
    x_round_tripped = transform.forward(y)
    tf.debugging.assert_equal(x_round_tripped["flux"], x["flux"])
    tf.debugging.assert_equal(x_round_tripped["sfc_down"], x["sfc_down"])
