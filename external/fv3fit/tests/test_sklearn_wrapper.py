import numpy as np
from sklearn.linear_model import LinearRegression
import sklearn.dummy
import unittest.mock
import pytest
import xarray as xr

from fv3fit.sklearn._wrapper import (
    RegressorEnsemble,
    pack,
    SklearnWrapper,
    TriggeredRegressor,
)
from fv3fit._shared.scaler import ManualScaler


def test_flatten():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a})

    ans = pack(ds, sample_dim)[0]
    assert ans.shape == (nz, 2 * nx * ny)


def test_flatten_1d_input():
    x = np.ones((3, 4, 5))
    shape = (3, 4, 5)
    dims = "x y z".split()
    sample_dim = "z"

    nx, ny, nz = shape

    a = xr.DataArray(x, dims=dims)
    ds = xr.Dataset({"a": a, "b": a.isel(x=0, y=0)})

    ans = pack(ds, sample_dim)[0]
    assert ans.shape == (nz, nx * ny + 1)


def test_flatten_same_order():
    nx, ny = 10, 4
    x = xr.DataArray(np.arange(nx * ny).reshape((nx, ny)), dims=["sample", "feature"])

    ds = xr.Dataset({"a": x, "b": x.T})
    sample_dim = "sample"
    a = pack(ds[["a"]], sample_dim)[0]
    b = pack(ds[["b"]], sample_dim)[0]

    np.testing.assert_allclose(a, b)


@pytest.fixture
def test_regressor_ensemble():
    base_regressor = LinearRegression()
    ensemble_regressor = RegressorEnsemble(base_regressor)
    num_batches = 3
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    for i in range(num_batches):
        ensemble_regressor.fit(X, y)
    return ensemble_regressor


def test_ensemble_fit(test_regressor_ensemble):
    regressor_ensemble = test_regressor_ensemble
    assert regressor_ensemble.n_estimators == 3
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    regressor_ensemble.fit(X, y)
    # test that .fit appends a new regressor
    assert regressor_ensemble.n_estimators == 4
    # test that new regressors are actually fit and not empty base regressor
    assert len(regressor_ensemble.regressors[-1].coef_) > 0


def _get_sklearn_wrapper(scale_factor=None, dumps_returns: bytes = b"HEY!"):
    model = unittest.mock.Mock()
    model.regressors = []
    model.base_regressor = unittest.mock.Mock()
    model.predict.return_value = np.array([[1.0]])
    model.dumps.return_value = dumps_returns

    if scale_factor:
        scaler = ManualScaler(np.array([scale_factor]))
    else:
        scaler = None

    return SklearnWrapper(
        sample_dim_name="sample",
        input_variables=["x"],
        output_variables=["y"],
        model=model,
        target_scaler=scaler,
    )


def test_SklearnWrapper_fit_predict_scaler(scale=2.0):
    wrapper = _get_sklearn_wrapper(scale)
    dims = ["sample", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(data)

    output = wrapper.predict(data)
    assert pytest.approx(1 / scale) == output["y"].item()


def test_fitting_SklearnWrapper_does_not_fit_scaler():
    """SklearnWrapper should use pre-computed scaling factors when fitting data
    
    In other words, calling the .fit method of wrapper should not call the
    .fit its scaler attribute.
    """

    model = unittest.mock.Mock()
    scaler = unittest.mock.Mock()

    wrapper = SklearnWrapper(
        sample_dim_name="sample",
        input_variables=["x"],
        output_variables=["y"],
        model=model,
        target_scaler=scaler,
    )

    dims = ["sample", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(data)
    scaler.fit.assert_not_called()


@pytest.mark.parametrize(
    "scale_factor", [2.0, None],
)
def test_SklearnWrapper_serialize_predicts_the_same(tmpdir, scale_factor):

    # Setup wrapper
    if scale_factor:
        scaler = ManualScaler(np.array([scale_factor]))
    else:
        scaler = None
    model = RegressorEnsemble(base_regressor=LinearRegression())
    wrapper = SklearnWrapper(
        sample_dim_name="sample",
        input_variables=["x"],
        output_variables=["y"],
        model=model,
        target_scaler=scaler,
    )

    # setup input data
    dims = ["sample", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(data)

    # serialize/deserialize
    path = str(tmpdir)
    wrapper.dump(path)

    loaded = wrapper.load(path)
    xr.testing.assert_equal(loaded.predict(data), wrapper.predict(data))


def test_SklearnWrapper_serialize_fit_after_load(tmpdir):
    model = RegressorEnsemble(base_regressor=LinearRegression())
    wrapper = SklearnWrapper(
        sample_dim_name="sample",
        input_variables=["x"],
        output_variables=["y"],
        model=model,
        target_scaler=None,
    )

    # setup input data
    dims = ["sample", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(data)

    # serialize/deserialize
    path = str(tmpdir)
    wrapper.dump(path)

    # fit loaded model
    loaded = wrapper.load(path)
    loaded.fit(data)

    assert len(loaded.model.regressors) == 2


def test_TriggeredRegressor_predict():
    classifier = sklearn.dummy.DummyClassifier()
    regressor = sklearn.dummy.DummyRegressor()

    n = 10

    X = np.zeros((1, n))
    y = np.zeros((1, 2 * n))
    label = np.zeros((1, 1))

    classifier.fit(X, label)
    regressor.fit(X, y)

    ds = xr.Dataset({"a": (["sample", "z"], X), "b": (["sample", "z"], y[:, :n])})

    model = TriggeredRegressor(
        classifier,
        regressor,
        sample_dim_name="sample",
        regressor_input_variables=["a"],
        classifier_input_variables=["a"],
        output_variables=["a", "b"],
    )

    out = model.predict(ds)
    assert isinstance(out, xr.Dataset)
    assert out["a"].shape[1] == n
    assert out["b"].shape[1] == n
