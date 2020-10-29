from typing import Dict
import yaml
import f90nml
from fv3fit.keras import get_model_class
from fv3fit.sklearn import SklearnWrapper, TriggeredRegressor
from fv3fit._shared import Predictor

FV3CONFIG_FILENAME = "fv3config.yml"


def get_config() -> Dict:
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    return config


def get_namelist() -> f90nml.Namelist:
    return f90nml.read("input.nml")


def get_ml_model(config: Dict) -> Predictor:
    model_class = _get_ml_model_class(config)
    return model_class.load(config["model"])


def _get_ml_model_class(config):
    model_type = config.get("model_type", "scikit_learn")
    if model_type == "keras":
        keras_model_type = config.get("model_loader_kwargs", {}).get(
            "keras_model_type", "DenseModel"
        )
        model_class = get_model_class(keras_model_type)
    elif model_type == "scikit_learn":
        model_class = SklearnWrapper
    elif model_type == "triggered_sklearn":
        return TriggeredRegressor
    else:
        raise ValueError(
            "Valid model type values include 'scikit_learn' and "
            f"'keras'; received {model_type}."
        )
    return model_class
