import io
import numpy as np
from sklearn.linear_model import LinearRegression
import unittest.mock
import pytest
import xarray as xr
import joblib

from fv3fit.sklearn._wrapper import RegressorEnsemble, pack, SklearnWrapper
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


def _get_sklearn_wrapper(scale_factor, dumps_returns: bytes = b"HEY!"):
    model = unittest.mock.Mock()
    model.regressors = []
    model.predict.return_value = np.array([[1.0]])
    model.dumps.return_value = dumps_returns

    scaler = ManualScaler(np.array([scale_factor]))

    return SklearnWrapper(
        sample_dim_name="z",
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


def test_SklearnWrapper_fit_scaler():
    scale = 2.0
    wrapper = _get_sklearn_wrapper(scale)
    dims = ["sample", "z"]
    data = xr.Dataset({"x": (dims, np.ones((1, 1))), "y": (dims, np.ones((1, 1)))})
    wrapper.fit(data)

    model: unittest.mock.Mock = wrapper.model

    x, y = model.fit.call_args[0]
    np.testing.assert_allclose(x, data.x.values)
    np.testing.assert_allclose(y, data.y.values * scale)


def test_SklearnWrapper_serialize(tmpdir):
    f = io.BytesIO()
    joblib.dump(["some", "list"], f)
    token = f.getvalue()

    wrapper = _get_sklearn_wrapper(scale_factor=1.0, dumps_returns=token)
    wrapper.dump(tmpdir)
    wrapper.load(tmpdir)
