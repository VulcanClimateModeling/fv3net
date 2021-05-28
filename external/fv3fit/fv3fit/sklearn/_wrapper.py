from typing import Mapping
import logging
import io
from copy import copy
import numpy as np
import xarray as xr
import pandas as pd
import fsspec
import joblib
from .._shared import (
    pack,
    unpack,
    Estimator,
    get_scaler,
    register_estimator,
)
from .._shared.config import RandomForestHyperparameters
from .. import _shared
from .._shared import scaler
import sklearn.base
import sklearn.ensemble

from typing import Optional, Iterable, Sequence
import yaml


def _multiindex_to_tuple(index: pd.MultiIndex) -> tuple:
    return list(index.names), list(index.to_list())


def _tuple_to_multiindex(d: tuple) -> pd.MultiIndex:
    names, list_ = d
    return pd.MultiIndex.from_tuples(list_, names=names)


@_shared.io.register("sklearn")
@register_estimator("sklearn", RandomForestHyperparameters)
@register_estimator("rf", RandomForestHyperparameters)
@register_estimator("random_forest", RandomForestHyperparameters)
@register_estimator("sklearn_random_forest", RandomForestHyperparameters)
class RandomForest(Estimator):
    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        hyperparameters: RandomForestHyperparameters,
    ):
        batch_regressor = RegressorEnsemble(
            sklearn.ensemble.RandomForestRegressor(
                n_jobs=hyperparameters.n_jobs,
                random_state=hyperparameters.random_state,
                n_estimators=hyperparameters.n_estimators,
                max_depth=hyperparameters.max_depth,
                min_samples_split=hyperparameters.min_samples_split,
                min_samples_leaf=hyperparameters.min_samples_leaf,
                max_features=hyperparameters.max_features,
            )
        )
        self._model_wrapper = SklearnWrapper(
            sample_dim_name,
            input_variables,
            output_variables,
            model=batch_regressor,
            scaler_type=hyperparameters.scaler_type,
            scaler_kwargs=hyperparameters.scaler_kwargs,
        )

    def fit(self, batches: Sequence[xr.Dataset]):
        return self._model_wrapper.fit(batches)

    def predict(self, features):
        return self._model_wrapper.predict(features)

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """
        self._model_wrapper.dump(path)

    @classmethod
    def load(cls, path: str) -> "SklearnWrapper":
        return SklearnWrapper.load(path)


class RegressorEnsemble:
    """Ensemble of regressors that are incrementally trained in batches

    """

    def __init__(
        self, base_regressor, regressors: Sequence[sklearn.base.BaseEstimator] = None,
    ) -> None:
        self.base_regressor = base_regressor
        self.regressors = regressors or []

    @property
    def n_estimators(self):
        return len(self.regressors)

    def fit(self, features, outputs):
        """ Adds a base regressor fit on features to the ensemble

        Args:
            features: numpy array of features
            outputs: numpy array of targets

        Returns:

        """
        new_regressor = copy(self.base_regressor)
        # each regressor needs different randomness
        if hasattr(new_regressor, "random_state"):
            new_regressor.random_state += len(self.regressors)
        new_regressor.fit(features, outputs)
        self.regressors.append(new_regressor)

    def predict(self, features):
        """

        Args:
            features: 2D numpy array of features to predict on

        Returns:
            2D numpy array of predictions with N rows corresponding to N input samples.
            Each row is the average ensemble prediction for that sample.
        """
        predictions = np.array(
            [regressor.predict(features) for regressor in self.regressors]
        )
        return np.mean(predictions, axis=0)

    def dumps(self) -> bytes:
        batch_regressor_components = {
            "regressors": self.regressors,
            "base_regressor": self.base_regressor,
        }
        f = io.BytesIO()
        joblib.dump(batch_regressor_components, f)
        return f.getvalue()

    @classmethod
    def loads(cls, b: bytes) -> "RegressorEnsemble":
        f = io.BytesIO(b)
        batch_regressor_components = joblib.load(f)
        regressors: Sequence[sklearn.base.BaseEstimator] = batch_regressor_components[
            "regressors"
        ]
        base_regressor = batch_regressor_components["base_regressor"]
        obj = cls(base_regressor=base_regressor, regressors=regressors)
        return obj


class SklearnWrapper(Estimator):
    """Wrap a SkLearn model for use with xarray

    """

    _PICKLE_NAME = "sklearn.pkl"
    _SCALER_NAME = "scaler.bin"
    _METADATA_NAME = "metadata.bin"

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
        model: RegressorEnsemble,
        parallel_backend: str = "threading",
        scaler_type: str = "standard",
        scaler_kwargs: Optional[Mapping] = None,
    ) -> None:
        """
        Initialize the wrapper

        Args:
            sample_dim_name: dimension over which samples are taken
            input_variables: list of input variables
            output_variables: list of output variables
            model: a scikit learn regression model
        """
        self._sample_dim_name = sample_dim_name
        self._input_variables = input_variables
        self._output_variables = output_variables
        self.model = model

        self.parallel_backend = parallel_backend
        self.scaler_type = scaler_type
        self.scaler_kwargs = scaler_kwargs or {}
        self.target_scaler: Optional[scaler.NormalizeTransform] = None

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def _fit_batch(self, data: xr.Dataset):
        # TODO the sample_dim can change so best to use feature dim to flatten
        x, _ = pack(data[self.input_variables], self.sample_dim_name)
        y, self.output_features_ = pack(
            data[self.output_variables], self.sample_dim_name
        )

        if self.target_scaler is None:
            self.target_scaler = self._init_target_scaler(data)

        y = self.target_scaler.normalize(y)
        self.model.fit(x, y)

    def _init_target_scaler(self, batch):
        return get_scaler(
            self.scaler_type,
            self.scaler_kwargs,
            batch,
            self._output_variables,
            self._sample_dim_name,
        )

    def fit(self, batches: Sequence[xr.Dataset]):
        logger = logging.getLogger("SklearnWrapper")
        for i, batch in enumerate(batches):
            logger.info(f"Fitting batch {i+1}/{len(batches)}")
            self._fit_batch(batch)
            logger.info(f"Batch {i+1} done fitting.")

    def predict(self, data):
        x, _ = pack(data[self.input_variables], self.sample_dim_name)
        with joblib.parallel_backend(self.parallel_backend, n_jobs=1):
            y = self.model.predict(x)

            if self.target_scaler is not None:
                y = self.target_scaler.denormalize(y)
            else:
                raise ValueError("Target scaler not present.")

        ds = unpack(y, self.sample_dim_name, self.output_features_)
        return ds.assign_coords({self.sample_dim_name: data[self.sample_dim_name]})

    def dump(self, path: str) -> None:
        """Dump data to a directory

        Args:
            path: a URL pointing to a directory
        """

        fs: fsspec.AbstractFileSystem = fsspec.get_fs_token_paths(path)[0]

        fs.makedirs(path, exist_ok=True)

        mapper = fs.get_mapper(path)
        mapper[self._PICKLE_NAME] = self.model.dumps()
        if self.target_scaler is not None:
            mapper[self._SCALER_NAME] = scaler.dumps(self.target_scaler).encode("UTF-8")

        metadata = [
            self.sample_dim_name,
            self.input_variables,
            self.output_variables,
            _multiindex_to_tuple(self.output_features_),
        ]

        mapper[self._METADATA_NAME] = yaml.safe_dump(metadata).encode("UTF-8")

    @classmethod
    def load(cls, path: str) -> "SklearnWrapper":
        """Load a model from a remote path"""
        mapper = fsspec.get_mapper(path)
        model = RegressorEnsemble.loads(mapper[cls._PICKLE_NAME])

        scaler_str = mapper.get(cls._SCALER_NAME, b"")
        scaler_obj: Optional[scaler.NormalizeTransform]
        if scaler_str:
            scaler_obj = scaler.loads(scaler_str)
        else:
            scaler_obj = None
        (
            sample_dim_name,
            input_variables,
            output_variables,
            output_features_dict_,
        ) = yaml.safe_load(mapper[cls._METADATA_NAME])

        output_features_ = _tuple_to_multiindex(output_features_dict_)

        obj = cls(sample_dim_name, input_variables, output_variables, model)
        obj.target_scaler = scaler_obj
        obj.output_features_ = output_features_

        return obj

    # these are here for backward compatibility with pre-unified API attribute names
    @property
    def input_variables(self):
        if hasattr(self, "_input_variables"):
            return self._input_variables
        elif hasattr(self, "input_vars_"):
            return self.input_vars_
        else:
            raise ValueError("Wrapped model version without input variables attribute.")

    @property
    def output_variables(self):
        if hasattr(self, "_input_variables"):
            return self._output_variables
        elif hasattr(self, "input_vars_"):
            return self.output_vars_
        else:
            raise ValueError(
                "Wrapped model version without output variables attribute."
            )

    @property
    def sample_dim_name(self):
        return getattr(self, "_sample_dim_name", "sample")
