import logging
import os
import tempfile

import numpy as np
import pytest
import xarray as xr
import yaml

import synth
from fv3fit import _shared as shared
import fv3fit
from offline_ml_diags._mapper import PredictionMapper
from offline_ml_diags._helpers import load_grid_info

from loaders import SAMPLE_DIM_NAME, batches, mappers
from offline_ml_diags.compute_diags import _average_metrics_dict, _compute_diagnostics

logger = logging.getLogger(__name__)

DIURNAL_VARS = [
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_pQ1",
    "column_integrated_pQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]
OUTPUT_NC_NAME = "diagnostics.nc"



@pytest.fixture
def training_data_diags_config(datadir_module):
    with open(
        os.path.join(str(datadir_module), "training_data_sources_config.yml"), "r"
    ) as f:
        yield yaml.safe_load(f)


def get_data_source_training_diags_config(config, data_source_name):
    source_config = config["sources"][data_source_name]
    return {
        "mapping_function": source_config["mapping_function"],
        "mapping_kwargs": source_config.get("mapping_kwargs", {}),
    }


def _nudging_train_config(datadir_module):
    with open(
        os.path.join(str(datadir_module), "train_sklearn_model_nudged_source.yml"), "r"
    ) as f:
        config = yaml.safe_load(f)
    return shared.ModelTrainingConfig("nudging_data_path", **config)


@pytest.fixture
def nudging_train_config(datadir_module):
    return _nudging_train_config(datadir_module)


def _fine_res_train_config(datadir_module):
    with open(
        os.path.join(str(datadir_module), "train_sklearn_model_fineres_source.yml"), "r"
    ) as f:
        config = yaml.safe_load(f)
    return shared.ModelTrainingConfig("fine_res_data_path", **config)


@pytest.fixture
def fine_res_train_config(datadir_module):
    return _fine_res_train_config(datadir_module)


@pytest.fixture
def data_source_train_config(data_source_name, datadir_module):
    if data_source_name == "nudging_tendencies":
        data_source_train_config = _nudging_train_config(datadir_module)
    elif data_source_name == "fine_res_apparent_sources":
        data_source_train_config = _fine_res_train_config(datadir_module)
    else:
        raise NotImplementedError()
    return data_source_train_config


class MockSklearnWrappedModel(fv3fit.Predictor):
    def __init__(self, input_vars, output_vars):
        self.input_variables = input_vars
        self.output_variables = output_vars
        self.sample_dim_name = "sample"

    def predict(self, ds_stacked, sample_dim=SAMPLE_DIM_NAME):
        ds_pred = xr.Dataset()
        for output_var in self.output_variables:
            feature_vars = [ds_stacked[var] for var in self.input_variables]
            mock_prediction = sum(feature_vars)
            ds_pred[output_var] = mock_prediction
        return ds_pred

    def load(self, *args, **kwargs):
        pass

    def dump(self, path):
        pass


input_vars = ("air_temperature", "specific_humidity")
output_vars = ("dQ1", "dQ2")


@pytest.fixture
def mock_model():
    return MockSklearnWrappedModel(input_vars, output_vars)


@pytest.fixture
def data_source_offline_config(
    data_source_name, datadir_module, C48_SHiELD_diags_dataset_path
):
    if data_source_name == "nudging_tendencies":
        with open(
            os.path.join(str(datadir_module), "train_sklearn_model_nudged_source.yml"),
            "r",
        ) as f:
            config = yaml.safe_load(f)
            config["batch_kwargs"]["mapping_kwargs"][
                "shield_diags_url"
            ] = C48_SHiELD_diags_dataset_path
        return config
    elif data_source_name == "fine_res_apparent_sources":
        with open(
            os.path.join(str(datadir_module), "train_sklearn_model_fineres_source.yml"),
            "r",
        ) as f:
            config = yaml.safe_load(f)
            config["batch_kwargs"]["mapping_kwargs"][
                "shield_diags_url"
            ] = C48_SHiELD_diags_dataset_path
            return config
    else:
        raise NotImplementedError()


@pytest.fixture
def prediction_mapper(
    mock_model, data_source_name, data_source_path, data_source_offline_config,
):

    base_mapping_function = getattr(
        mappers, data_source_offline_config["batch_kwargs"]["mapping_function"]
    )
    base_mapper = base_mapping_function(
        data_source_path,
        **data_source_offline_config["batch_kwargs"].get("mapping_kwargs", {}),
    )
    grid = load_grid_info(res="c8_random_values")
    prediction_mapper = PredictionMapper(base_mapper, mock_model, variables, grid=grid)

    return prediction_mapper


timesteps = ["20160801.001500", "20160801.003000"]
variables = [
    "air_temperature",
    "specific_humidity",
    "dQ1",
    "dQ2",
    "pQ1",
    "pQ2",
    "pressure_thickness_of_atmospheric_layer",
    "net_heating",
    "net_precipitation",
    "area",
    "land_sea_mask",
]


@pytest.fixture
def diagnostic_batches(prediction_mapper, data_source_offline_config):

    data_source_offline_config["batch_kwargs"]["timesteps"] = timesteps
    data_source_offline_config["variables"] = variables
    del data_source_offline_config["batch_kwargs"]["mapping_function"]
    del data_source_offline_config["batch_kwargs"]["mapping_kwargs"]
    diagnostic_batches = batches.batches_from_mapper(
        prediction_mapper,
        data_source_offline_config["variables"],
        training=False,
        needs_grid=False,
        **data_source_offline_config["batch_kwargs"],
    )
    return diagnostic_batches


@pytest.mark.regression
def test_compute_offline_diags(
    offline_diags_reference_schema,
    diagnostic_batches,
    grid_dataset,
    data_source_offline_config,
):
    ds_diagnostics, ds_diurnal, ds_metrics = _compute_diagnostics(
        diagnostic_batches,
        grid_dataset,
        predicted_vars=data_source_offline_config["output_variables"],
    )

    # convert metrics to dict
    metrics = _average_metrics_dict(ds_metrics)
    for var in DIURNAL_VARS:
        assert "local_time_hr" in ds_diurnal[var].dims
        for dim in ds_diurnal[var].dims:
            assert dim in ["local_time_hr", "derivation", "surface_type"]

    assert isinstance(metrics, dict)
    assert len(metrics) == 32
    for metric, metric_entry in metrics.items():
        assert isinstance(metric, str)
        assert isinstance(metric_entry, dict)
        for metric_key, metric_value in metric_entry.items():
            assert isinstance(metric_key, str)
            assert isinstance(metric_value, (float, np.float32))



def test_offline_diags_integration(
    diagnostic_batches,
    grid_dataset,
    data_source_offline_config,
):
    """
    Test the bash endpoint for training the model produces the expected output files.
    """
    subprocess.check_call(
        [
            "python",
            "-m",
            "offline_ml_diags.compute_diags",
            data_source_path,
            train_config_filename,
            tmp_path,
        ]
    )
    required_names = ["model_data", "training_config.yml"]
    missing_names = set(required_names).difference(os.listdir(tmp_path))
    assert len(missing_names) == 0