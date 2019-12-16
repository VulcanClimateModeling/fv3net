from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.base import BaseEstimator


def remove(dims, sample_dim):
    return tuple([dim for dim in dims if dim != sample_dim])


def unused_name(old_names):
    # should not conflict with existing name
    # a random string that no-one will ever use
    return "dadf3q32d9a09cf"


def _flatten(data: xr.Dataset, sample_dim) -> np.ndarray:
    feature_dim_name = unused_name(data.dims)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return stacked.transpose(sample_dim, feature_dim_name)


@dataclass
class BaseXarrayEstimator:
    def fit(
            self, input_vars: tuple, output_vars: tuple, sample_dim: str,
            data: xr.Dataset
    ):
        """
        Args:
            input_vars: list of input variables
            output_vars: list of output_variables
            sample_dim: dimension over which samples are taken
            data: xarray Dataset with dimensions (sample_dim, *)

        Returns:
            fitted model
        """
        raise NotImplementedError

    def predict(self, data: xr.Dataset, sample_dim: str) -> xr.Dataset:
        """
        Make a prediction

        Args:
            data: xarray Dataset with the same feature dimensions as trained
              data
            sample_dim: dimension along which "samples" are defined. This could be
              inferred, but explicity is not terrible.
        Returns:
            prediction:
        """
        raise NotImplementedError


class SklearnWrapper(BaseXarrayEstimator):
    """Wrap a SkLearn model for use with xarray

    """

    def __init__(self, model: BaseEstimator):
        """

        Args:
            model: a scikit learn regression model
        """
        self.model = model
        if 'n_estimators' in self.model.__dict__:
            self.n_estimators_per_batch = self.model.n_estimators
        elif hasattr(self.model, 'estimator') and \
                'n_estimators' in self.model.estimator.__dict__:
            self.n_estimators_per_batch = self.model.estimator.n_estimators

    def fit(self, features, targets):
        self.model.fit(features, targets)

    def fit_xarray(
            self, input_vars: tuple, output_vars: tuple, sample_dim: str,
            data: xr.Dataset
    ):

        self.input_vars_ = input_vars
        self.output_vars_ = output_vars
        self.feature_dims_ = remove(data.dims, sample_dim)
        inputs = _flatten(data[input_vars], sample_dim).values
        outputs = _flatten(data[output_vars], sample_dim)

        self.output_features_dim_name_ = [
            dim for dim in outputs.dims if dim != sample_dim
        ][0]
        self.output_features_ = outputs.indexes[self.output_features_dim_name_]
        self.model.fit(inputs, outputs.values)

    def add_new_batch_estimators(self):
        if 'n_estimators' in self.model.__dict__:
            self.model.n_estimators += self.n_estimators_per_batch
        elif hasattr(self.model, 'estimator') and \
             'n_estimators' in self.model.estimator.__dict__:
            self.model.set_params(
                estimator__n_estimators = \
                    self.model.estimator.n_estimators + self.n_estimators_per_batch)
        else:
            raise ValueError("Cannot add more estimators to model. Check that model is"
                             "either sklearn RandomForestRegressor "
                             "or MultiOutputRegressor.")

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def predict(self, features):
        return self.model.predict(features)

    def predict_xrarray(self, data, sample_dim):
        inputs = _flatten(data[self.input_vars_], sample_dim).values
        numpy = self.model.predict(inputs)
        ds = xr.DataArray(
            numpy,
            dims=[sample_dim, "feature"],
            coords={sample_dim: inputs[sample_dim], "feature": self.output_features_},
        )

        return ds.to_unstacked_dataset("feature")

