from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
import loaders
import fv3fit
import numpy as np
import tempfile
import subprocess
import os
import copy

from fv3fit.keras._training import set_random_seed


logger = logging.getLogger(__name__)


@pytest.fixture(params=["DenseModel"])
def model_type(request) -> str:
    return request.param


@pytest.fixture(params=["mse"])
def loss(request) -> str:
    return request.param


@pytest.fixture(params=[{"width": 4, "depth": 3}])
def hyperparameters(request, model_type, loss) -> dict:
    if model_type == "DenseModel":
        hyperparameters = request.param
        if loss:
            hyperparameters["loss"] = loss
        return hyperparameters
    else:
        raise NotImplementedError(model_type)


@pytest.fixture
def model(
    hyperparameters: dict,
    train_config,
) -> fv3fit.Estimator:
    fit_kwargs = hyperparameters.pop("fit_kwargs", {})
    return fv3fit.keras.get_model(
        train_config.model_type,
        loaders.SAMPLE_DIM_NAME,
        train_config.input_variables,
        train_config.output_variables,
        **train_config.hyperparameters,
        **fit_kwargs,
    )


def test_reproducibility(
    train_config,
    training_batches: Sequence[xr.Dataset],
):
    batch_dataset_test = training_batches[0]
    fit_kwargs = {"batch_size": 384, "validation_samples": 384}
    set_random_seed(0)
    model_0 = fv3fit.keras.get_model(
        "DenseModel",
        loaders.SAMPLE_DIM_NAME,
        train_config.input_variables,
        train_config.output_variables,
        fit_kwargs=copy.deepcopy(fit_kwargs),
        **train_config.hyperparameters,
    )
    model_0.fit(training_batches)
    result_0 = model_0.predict(batch_dataset_test)

    set_random_seed(0)
    model_1 = fv3fit.keras.get_model(
        "DenseModel",
        loaders.SAMPLE_DIM_NAME,
        train_config.input_variables,
        train_config.output_variables,
        fit_kwargs=copy.deepcopy(fit_kwargs),
        **train_config.hyperparameters,
    )
    model_1.fit(training_batches)
    result_1 = model_1.predict(batch_dataset_test)

    xr.testing.assert_allclose(result_0, result_1, rtol=1e-03)


def test_training(
    model: fv3fit.Estimator,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    model.fit(training_batches)
    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    validate_dataset_result(result, batch_dataset, output_variables)


def test_dump_and_load_before_training(
    model: fv3fit.Estimator,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        model = model.__class__.load(tmpdir)
    model.fit(training_batches)
    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    validate_dataset_result(result, batch_dataset, output_variables)


def validate_dataset_result(
    result: xr.Dataset, batch_dataset: xr.Dataset, output_variables: Iterable[str]
):
    """
    Use assertions to test whether the predicted output dataset metadata matches
    metadata from a reference, for the given variable names. Also checks output values
    are present.
    """
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == batch_dataset[varname].shape, varname
        assert np.sum(np.isnan(result[varname].values)) == 0


def test_dump_and_load_maintains_prediction(
    model: fv3fit.Estimator,
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
):
    model.fit(training_batches)
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        loaded_model = model.__class__.load(tmpdir)
    batch_dataset = training_batches[0]
    loaded_result = loaded_model.predict(batch_dataset)
    validate_dataset_result(loaded_result, batch_dataset, output_variables)
    original_result = model.predict(batch_dataset)
    xr.testing.assert_equal(loaded_result, original_result)


hyperparams_with_fit_kwargs = {
    "width": 4,
    "depth": 3,
    "fit_kwargs": {"batch_size": 100, "validation_samples": 384},
}


@pytest.mark.parametrize(
    "hyperparameters, validation_timesteps",
    [
        (hyperparams_with_fit_kwargs, ["20160801.003000"]),
        (hyperparams_with_fit_kwargs, None),
    ],
    indirect=["hyperparameters", "validation_timesteps"],
)
def test_training_integration(
    hyperparameters,
    validation_timesteps,
    data_and_config,
    tmp_path: str,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """
    data_source_path, train_config_filename = data_and_config
    subprocess.check_call(
        [
            "python",
            "-m",
            "fv3fit.train",
            data_source_path,
            train_config_filename,
            tmp_path,
        ]
    )
    required_names = ["model_data", "training_config.yml"]
    missing_names = set(required_names).difference(os.listdir(tmp_path))
    assert len(missing_names) == 0


@pytest.mark.parametrize(
    "loss, hyperparameters, expected_loss",
    (
        pytest.param("mae", {}, "mae", id="specified_loss"),
        pytest.param(None, {}, "mse", id="default_loss"),
    ),
    indirect=["loss", "hyperparameters"],
)
def test_dump_and_load_loss_info(loss, hyperparameters, expected_loss, model):
    with tempfile.TemporaryDirectory() as tmpdir:
        model.dump(tmpdir)
        model_loaded = model.__class__.load(tmpdir)
    assert model_loaded._loss == expected_loss
