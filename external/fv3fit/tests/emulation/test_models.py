import tempfile
from os.path import join
from fv3fit.emulation.keras import save_model

import numpy as np
import pytest
import tensorflow as tf
from fv3fit._shared import SliceConfig
from fv3fit.emulation.layers import ArchitectureConfig
from fv3fit.emulation.layers.architecture import _ARCHITECTURE_KEYS
from fv3fit.emulation.models import MicrophysicsConfig, transform_model
from fv3fit.emulation.transforms.transforms import Identity


def _get_data(shape):

    num = int(np.prod(shape))
    return np.arange(num).reshape(shape).astype(np.float32)


def _get_tensor(shape):
    return tf.convert_to_tensor(_get_data(shape))


def test_Config():

    config = MicrophysicsConfig(
        input_variables=["dummy_in"], direct_out_variables=["dummy_out"]
    )
    assert config.input_variables == ["dummy_in"]
    assert config.direct_out_variables == ["dummy_out"]


def test_Config_from_dict():
    config = MicrophysicsConfig.from_dict(
        dict(input_variables=["dummy_in"], direct_out_variables=["dummy_out"],)
    )
    assert config.input_variables == ["dummy_in"]
    assert config.direct_out_variables == ["dummy_out"]


def test_Config_from_dict_selection_map_sequences():
    config = MicrophysicsConfig.from_dict(
        dict(selection_map=dict(dummy=dict(start=0, stop=2, step=1)))
    )
    assert config.selection_map["dummy"].slice == slice(0, 2, 1)


def test_Config_asdict():
    sl1_kwargs = dict(start=0, stop=10, step=2)
    sl2_kwargs = dict(start=None, stop=25, step=None)
    sel_map = dict(
        dummy_in=SliceConfig(**sl1_kwargs), dummy_out=SliceConfig(**sl2_kwargs)
    )

    original = MicrophysicsConfig(
        input_variables=["dummy_in"],
        direct_out_variables=["dummy_out"],
        selection_map=sel_map,
    )

    config_d = original.asdict()
    assert config_d["selection_map"]["dummy_in"] == sl1_kwargs
    assert config_d["selection_map"]["dummy_out"] == sl2_kwargs

    result = MicrophysicsConfig.from_dict(config_d)
    assert result == original


def test_Config_build():

    config = MicrophysicsConfig(
        input_variables=["dummy_in"], direct_out_variables=["dummy_out"],
    )

    data = _get_data((20, 5))
    m = {"dummy_in": data, "dummy_out": data}
    model = config.build(m)
    output = model(m)
    assert set(output) == {"dummy_out"}


@pytest.mark.parametrize("arch", _ARCHITECTURE_KEYS)
@pytest.mark.slow
def test_MicrophysicConfig_model_save_reload(arch):

    config = MicrophysicsConfig(
        input_variables=["field_input"],
        direct_out_variables=["field_output"],
        architecture=ArchitectureConfig(name=arch),
    )

    nlev = 15
    data = tf.random.normal((10, nlev))
    sample = {"field_input": data, "field_output": data}

    model = config.build(sample)

    expected = model(sample)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = join(tmpdir, "model.tf")
        model.save(model_path, save_format="tf")
        reloaded = tf.keras.models.load_model(model_path, compile=False)

    result = reloaded(sample)
    np.testing.assert_allclose(
        expected["field_output"], result["field_output"], rtol=2e-5
    )


@pytest.mark.slow
def test_RNN_downward_dependence():

    config = MicrophysicsConfig(
        input_variables=["field_input"],
        direct_out_variables=["field_output"],
        architecture=ArchitectureConfig(name="rnn-v1", kwargs=dict(channels=16)),
    )

    nlev = 15
    data = tf.random.normal((10, nlev))
    sample = {"field_input": data, "field_output": data}
    profile = data[0:1]

    model = config.build(sample)

    with tf.GradientTape() as g:
        g.watch(profile)
        output = model(profile)

    jacobian = g.jacobian(output["field_output"], profile)[0, :, 0]

    assert jacobian.shape == (nlev, nlev)
    for output_level in range(nlev):
        for input_level in range(nlev):
            sensitivity = jacobian[output_level, input_level]
            if output_level > input_level and sensitivity != 0:
                raise ValueError("Downwards dependence violated")


def test_save_and_reload_transformed_model(tmpdir):
    inputs = {"a": tf.keras.Input(10, name="a")}
    outputs = {"out": tf.keras.layers.Lambda(lambda x: x, name="out")(inputs["a"])}
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    transformed_model = transform_model(model, Identity, inputs)

    input = {"a": tf.ones((1, 10))}
    output = {"out": tf.ones((1, 10))}
    data = {**input, **output}
    model(data)
    path = str(tmpdir) + "/model.tf"
    save_model(transformed_model, str(tmpdir))
    loaded = tf.keras.models.load_model(path)
    # will often complain since ``input`` is missing the "a_out" field which was
    # passed to ``model`` above.
    loaded.predict(input)


def test_transform_model_input_names():
    class MockTransform:
        def forward(self, x):
            y = {**x}
            y["transform"] = x["a"]
            return y

        def backward(self, x):
            y = {**x}
            y["out"] = x["transform_out"]
            # try to grab input field
            x["a"]
            return y

    original_inputs = {"a": tf.keras.Input(10, name="a")}
    # make model in transformed space
    transform_input = tf.keras.Input(10, name="transform")
    transform_output = tf.keras.layers.Lambda(lambda x: x, name="transform_out")(
        transform_input
    )
    model = tf.keras.Model(inputs=[transform_input], outputs=[transform_output])
    transformed_model = transform_model(model, MockTransform(), original_inputs)
    # only includes untransformed inputs
    assert set(transformed_model.input_names) == {"a"}
    # includes both transformed and untransformed outpus
    assert set(transformed_model.output_names) == {"out", "transform_out"}


@pytest.mark.xfail
@pytest.mark.slow
def test_saved_model_jacobian():
    """
    SimpleRNN saving prevents jacobian calculation due to some internal
    metadata missing after loading. Perhaps a tensorflow version upgrade
    fixes?
    """

    config = MicrophysicsConfig(
        input_variables=["field_input"],
        direct_out_variables=["field_output"],
        architecture=ArchitectureConfig(
            name="rnn-v1-shared-weights", kwargs=dict(channels=16)
        ),
    )

    nlev = 15
    data = tf.random.normal((10, nlev))
    sample = {"field_input": data, "field_output": data}
    profile = data[0:1]

    model = config.build(sample)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = join(tmpdir, "model.tf")
        model.save(save_path, save_format="tf")
        loaded_model = tf.keras.models.load_model(save_path)

    with tf.GradientTape() as g:
        g.watch(profile)
        output = loaded_model(profile)

    assert g.jacobian(output["field_output"], profile)


def test_Microphysics_unscaled_outputs():
    builder = MicrophysicsConfig(unscaled_outputs=["a"])
    assert builder.output_variables == ["a"]
