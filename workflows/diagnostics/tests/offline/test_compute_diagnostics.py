import numpy as np
import pytest
import xarray as xr

from fv3net.diagnostics.offline.compute_diagnostics import (
    weighted_average,
    _snap_mask_to_type,
    _snap_net_precipitation_to_type,
    _conditional_average,
)

da = xr.DataArray(np.arange(1.0, 5.0), dims=["z"])
da_nans = xr.DataArray(np.full((4,), np.nan), dims=["z"])
ds = xr.Dataset({"a": da})
weights = xr.DataArray([0.5, 0.5, 1, 1], dims=["z"])
weights_nans = xr.DataArray(np.full((4,), np.nan), dims=["z"])


@pytest.mark.parametrize(
    "da,weights,dims,expected",
    [
        (da, weights, "z", xr.DataArray(17.0 / 6.0)),
        (ds, weights, "z", xr.Dataset({"a": xr.DataArray(17.0 / 6.0)})),
        (da_nans, weights, "z", xr.DataArray(0.0)),
        (da, weights_nans, "z", xr.DataArray(np.nan)),
    ],
)
def test_weighted_average(da, weights, dims, expected):
    xr.testing.assert_allclose(weighted_average(da, weights, dims), expected)


def test_weighted_averaged_no_dims():

    da = xr.DataArray([[[np.arange(1.0, 5.0)]]], dims=["tile", "y", "x", "z"])
    weights = xr.DataArray([[[[0.5, 0.5, 1, 1]]]], dims=["tile", "y", "x", "z"])
    expected = xr.DataArray(np.arange(1.0, 5.0), dims=["z"])

    xr.testing.assert_allclose(weighted_average(da, weights), expected)


enumeration = {1: "land", 0: "sea", 2: "sea"}


@pytest.mark.parametrize(
    "float_mask,enumeration,atol,expected",
    [
        pytest.param(
            xr.DataArray([1.0, 0.0, 2.0], dims=["x"]),
            enumeration,
            1e-7,
            xr.DataArray(["land", "sea", "sea"], dims=["x"]),
            id="exact",
        ),
        pytest.param(
            xr.DataArray([1.0000001, 0.0], dims=["x"]),
            enumeration,
            1e-7,
            xr.DataArray(["land", "sea"], dims=["x"]),
            id="within_atol",
        ),
        pytest.param(
            xr.DataArray([1.0001, 0.0], dims=["x"]),
            enumeration,
            1e-7,
            xr.DataArray([np.nan, "sea"], dims=["x"]),
            id="outside_atol",
        ),
    ],
)
def test__snap_mask_to_type(float_mask, enumeration, atol, expected):
    xr.testing.assert_equal(_snap_mask_to_type(float_mask, enumeration, atol), expected)


@pytest.mark.parametrize(
    "net_precipitation,type_names,expected",
    [
        pytest.param(
            xr.DataArray([-1.0, 1.0], dims=["x"]),
            None,
            xr.DataArray(
                ["negative_net_precipitation", "positive_net_precipitation"], dims=["x"]
            ),
            id="positive_and_negative",
        ),
        pytest.param(
            xr.DataArray([-1.0, 0.0], dims=["x"]),
            None,
            xr.DataArray(
                ["negative_net_precipitation", "positive_net_precipitation"], dims=["x"]
            ),
            id="positive_and_zero",
        ),
        pytest.param(
            xr.DataArray([-1.0, 1.0], dims=["x"]),
            {"negative": "negative", "positive": "positive"},
            xr.DataArray(["negative", "positive"], dims=["x"]),
            id="custom_names",
        ),
    ],
)
def test__snap_net_precipitation_to_type(net_precipitation, type_names, expected):
    xr.testing.assert_equal(
        _snap_net_precipitation_to_type(net_precipitation, type_names), expected
    )


ds = xr.Dataset(
    {"a": xr.DataArray([[[np.arange(1.0, 5.0)]]], dims=["z", "tile", "y", "x"])}
)
surface_type_da = xr.DataArray(
    [[[["sea", "land", "land", "land"]]]], dims=["z", "tile", "y", "x"]
)
area = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims=["x"])


@pytest.mark.parametrize(
    "ds,surface_type_da,surface_type,area,expected",
    [
        (
            ds,
            surface_type_da,
            "sea",
            area,
            xr.Dataset({"a": xr.DataArray([1.0], dims=["z"])}),
        ),
        (
            ds,
            surface_type_da,
            "land",
            area,
            xr.Dataset({"a": xr.DataArray([3.0], dims=["z"])}),
        ),
    ],
)
def test__conditional_average(ds, surface_type_da, surface_type, area, expected):

    average = _conditional_average(ds, surface_type_da, surface_type, area)
    xr.testing.assert_allclose(average, expected)
