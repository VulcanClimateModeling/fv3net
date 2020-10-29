import abc
import io
from copy import copy
import numpy as np
import xarray as xr
import pandas as pd
import fsspec
import joblib
from .. import _shared
from .._shared import pack, unpack, Predictor
from .._shared import scaler
import sklearn.base

from typing import Optional, Iterable, Sequence
import yaml
from dataclasses import dataclass


def _multiindex_to_tuple(index: pd.MultiIndex) -> tuple:
    return list(index.names), list(index.to_list())


def _tuple_to_multiindex(d: tuple) -> pd.MultiIndex:
    names, list_ = d
    return pd.MultiIndex.from_tuples(list_, names=names)


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


class BaseXarrayEstimator(Predictor):
    """Abstract base class for an estimator wich works with xarray datasets.
    Subclasses Predictor abstract base class but adds `fit` method taking in
    xarray dataset.
    """

    def __init__(
        self,
        sample_dim_name: str,
        input_variables: Iterable[str],
        output_variables: Iterable[str],
    ):
        """
        Args:
            sample_dim_name: dimension over which samples are taken
            input_variables: list of input variables
            output_variables: list of output variables
        """
        super().__init__(sample_dim_name, input_variables, output_variables)

    @abc.abstractmethod
    def fit(self, data: xr.Dataset) -> None:
        """
        Args:
            data: xarray Dataset with dimensions (sample_dim, *)

        """
        raise NotImplementedError


@dataclass
class TriggeredRegressor(Predictor):
    classifier: sklearn.base.BaseEstimator
    regressor: sklearn.base.BaseEstimator
    sample_dim_name: str
    regressor_input_variables: Iterable[str]
    classifier_input_variables: Iterable[str]
    output_variables: Iterable[str]

    @property
    def regressor_x_packer(self) -> _shared.ArrayPacker:
        return _shared.ArrayPacker(self.sample_dim_name, self.regressor_input_variables)

    @property
    def classifier_x_packer(self) -> _shared.ArrayPacker:
        return _shared.ArrayPacker(
            self.sample_dim_name, self.classifier_input_variables
        )

    @property
    def input_variables(self):
        return list(self.regressor_input_variables) + list(
            self.classifier_input_variables
        )

    def predict(self, data):
        X_reg = self.regressor_x_packer.to_array(data)
        X_cl = self.classifier_x_packer.to_array(data)

        with joblib.parallel_backend("threading", n_jobs=1):
            labels = self.classifier.predict(X_cl).ravel().astype(bool)
            output = self.regressor.predict(X_reg) * labels.reshape((-1, 1))

        tendencies = np.split(output, len(self.output_variables), axis=1)
        data_vars = {
            key: ([self.sample_dim_name, "z"], tend)
            for key, tend in zip(self.output_variables, tendencies)
        }
        return xr.Dataset(data_vars, coords=data.coords)

    @staticmethod
    def load(path):
        import io

        fs = fsspec.get_mapper(path)
        reg = joblib.load(io.BytesIO(fs["regressor.pkl"]))
        clas = joblib.load(io.BytesIO(fs["nn.pkl"]))
        return TriggeredRegressor(
            clas["model"],
            reg["model"],
            sample_dim_name="sample",
            regressor_input_variables=reg["input_variables"],
            classifier_input_variables=clas["input_variables"],
            output_variables=reg["output_variables"],
        )


class SklearnWrapper(BaseXarrayEstimator):
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
        target_scaler: Optional[scaler.NormalizeTransform] = None,
        parallel_backend: str = "threading",
        n_jobs: int = 1,
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

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.target_scaler = target_scaler

    def __repr__(self):
        return "SklearnWrapper(\n%s)" % repr(self.model)

    def fit(self, data: xr.Dataset):
        # TODO the sample_dim can change so best to use feature dim to flatten
        x, _ = pack(data[self.input_variables], self.sample_dim_name)
        y, self.output_features_ = pack(
            data[self.output_variables], self.sample_dim_name
        )

        if self.target_scaler is not None:
            y = self.target_scaler.normalize(y)

        self.model.fit(x, y)

    def predict(self, data):
        x, _ = pack(data[self.input_variables], self.sample_dim_name)
        with joblib.parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            y = self.model.predict(x)

            if self.target_scaler is not None:
                y = self.target_scaler.denormalize(y)

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

        obj = cls(
            sample_dim_name, input_variables, output_variables, model, scaler_obj,
        )
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
