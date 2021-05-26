from typing import Iterable, Type, Union
import logging
import numpy as np
import os
import random
import tensorflow as tf
from .._shared.predictor import Estimator
from .._shared.config import get_keras_model

logger = logging.getLogger(__file__)

__all__ = ["get_model"]


# TODO: delete this and use the unified get_model instead
# when the tests are refactored to no longer depend on this function
def get_model(
    model_type: str,
    sample_dim_name: str,
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    **hyperparameters
) -> Estimator:
    """Initialize and return a Estimator instance.

    Args:
        model_type: the type of model to return
        input_variables: input variable names
        output_variables: output variable names
        **hyperparameters: other settings relevant to the model_type chosen

    Returns:
        model
    """
    return get_keras_model(model_type)(  # type: ignore
        sample_dim_name, input_variables, output_variables, **hyperparameters
    )


# TODO: merge this helper function with get_keras_model
def get_model_class(model_type: str) -> Type[Estimator]:
    """Returns a class implementing the Estimator interface corresponding to the model type.
    
    Args:
        model_type: the type of model

    Returns:
        model_class: a subclass of Estimator corresponding to the model type
    """
    return get_keras_model(model_type)


def set_random_seed(seed: Union[float, int] = 0):
    # https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    tf.random.set_seed(seed + 3)
