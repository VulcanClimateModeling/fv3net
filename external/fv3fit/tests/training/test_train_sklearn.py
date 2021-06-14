from typing import Iterable, Sequence
import xarray as xr
import pytest
import logging
from fv3fit._shared import _ModelTrainingConfig as ModelTrainingConfig
from fv3fit._shared.config import legacy_config_to_new_config
import numpy as np
import copy


from fv3fit.sklearn._random_forest import RandomForest

logger = logging.getLogger(__name__)


@pytest.fixture(params=["sklearn_random_forest"])
def model_type(request) -> str:
    return request.param


@pytest.fixture
def hyperparameters(model_type) -> dict:
    if model_type == "sklearn_random_forest":
        return {"max_depth": 4, "n_estimators": 2}
    else:
        raise NotImplementedError(model_type)


def test_training(
    training_batches: Sequence[xr.Dataset],
    output_variables: Iterable[str],
    train_config: ModelTrainingConfig,
):
    config = legacy_config_to_new_config(train_config)
    model = RandomForest(
        sample_dim_name="sample",
        input_variables=train_config.input_variables,
        output_variables=train_config.output_variables,
        hyperparameters=config.hyperparameters,
    )
    model.fit(training_batches)
    # This is the number of random forests in the ensemble, not the
    # number of total trees across the ensemble
    assert model._model_wrapper.model.n_estimators == 1

    # assert that the target scaler is fitted
    assert model._model_wrapper.target_scaler is not None

    batch_dataset = training_batches[0]
    result = model.predict(batch_dataset)
    missing_names = set(output_variables).difference(result.data_vars.keys())
    assert len(missing_names) == 0
    for varname in output_variables:
        assert result[varname].shape == batch_dataset[varname].shape, varname
        assert np.sum(np.isnan(result[varname].values)) == 0


def test_reproducibility(
    training_batches: Sequence[xr.Dataset], train_config: ModelTrainingConfig,
):
    batch_dataset = training_batches[0]
    train_config.hyperparameters["random_state"] = 0
    config = legacy_config_to_new_config(train_config)

    model_0 = RandomForest(
        sample_dim_name="sample",
        input_variables=train_config.input_variables,
        output_variables=train_config.output_variables,
        hyperparameters=config.hyperparameters,
    )
    model_0.fit(copy.deepcopy(training_batches))
    result_0 = model_0.predict(batch_dataset)

    model_1 = RandomForest(
        sample_dim_name="sample",
        input_variables=train_config.input_variables,
        output_variables=train_config.output_variables,
        hyperparameters=config.hyperparameters,
    )
    model_1.fit(copy.deepcopy(training_batches))
    result_1 = model_1.predict(batch_dataset)

    xr.testing.assert_allclose(result_0, result_1)
