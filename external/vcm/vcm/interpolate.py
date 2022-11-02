from typing import Callable, Union, Optional, TypeVar
import functools
import metpy.interpolate
import numpy as np
import xarray as xr
from scipy.spatial import KDTree

try:
    import mappm
except ModuleNotFoundError:
    mappm = None

import vcm.calc.thermo

import warnings

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="Interpolation point out of data bounds encountered",
)
# action=once does not work, so just ignoring the specific FutureWarning
warnings.filterwarnings(
    action="ignore",
    category=FutureWarning,
    message="``output_sizes`` should be given in the ``dask_gufunc_kwargs`` parameter. "
    "It will be removed as direct parameter in a future version.",
)

# for use in regridding values to the same vertical grid [Pa]
# These are taken to match the levels of the ERA-Interim reanalysis dataset
# https://psl.noaa.gov/data/gridded/data.erainterim.html
PRESSURE_GRID = xr.DataArray(
    [
        300.0,
        500.0,
        700.0,
        1000.0,
        2000.0,
        3000.0,
        5000.0,
        7000.0,
        10000.0,
        12500.0,
        15000.0,
        17500.0,
        20000.0,
        22500.0,
        25000.0,
        30000.0,
        35000.0,
        40000.0,
        45000.0,
        50000.0,
        55000.0,
        60000.0,
        65000.0,
        70000.0,
        75000.0,
        77500.0,
        80000.0,
        82500.0,
        85000.0,
        87500.0,
        90000.0,
        92500.0,
        95000.0,
        97500.0,
        100000.0,
    ],
    dims="pressure",
)

T = TypeVar("T", xr.DataArray, xr.Dataset)


def interpolate_to_pressure_levels(
    field: T,
    delp: xr.DataArray,
    levels: xr.DataArray = PRESSURE_GRID,
    dim: str = "pfull",
) -> T:
    """Regrid an atmospheric field to a fixed set of pressure levels

    Args:
        field: atmospheric quantity defined on hybrid vertical coordinates
        delp: pressure thickness of model layers in Pa. Must be broadcastable with
            ``da``
        dim: the vertical dimension name
        levels: 1D DataArray of output pressure levels

    Returns:
        the atmospheric quantity defined on ``pressure_levels``.
    """
    return interpolate_1d(
        levels, vcm.pressure_at_midpoint_log(delp, dim=dim), field, dim=dim,
    )


def interpolate_1d(
    xp: xr.DataArray, x: xr.DataArray, field: T, dim: Optional[str] = None,
) -> T:
    """Interpolates data with any shape over a specified axis.

    Wraps metpy.interpolate.interplolate_1d

    Args:
        xp: desired output levels.
        x: the original coordinate of ``field``. Must be increasing along the
            interpolating dimension.
        field: the quantity to be regridded. If a dataset, then all variables
            sharing dimensions with "x" will be interpolated, and other
            variables remain unchanged.
        dim: the dimension to interpolate over, only needed if xp is 1D. When
            ``xp`` is n-dimensional, the dimension to interpolate along is
            the one that differs with ``x``. For example, if xp.dims is ["x",
            "y_new"] and x.dims is ["x", "y"], then this function
            interpolates from "y" to "y_new".

    Returns:
        the quantity interpolated at the levels in ``xp``.

    See Also:
        https://unidata.github.io/MetPy/latest/api/generated/metpy.interpolate.interpolate_1d.html

    """
    if isinstance(field, xr.Dataset):
        data_vars = {}
        for v in field:
            if set(field[v].dims) >= set(x.dims):
                data_vars[v] = interpolate_1d(xp, x, field[v], dim=dim)
            else:
                data_vars[v] = field[v]
            data_vars[v].attrs = field[v].attrs
        return xr.Dataset(data_vars)
    elif isinstance(field, xr.DataArray):
        if xp.ndim == 1:
            if dim is None:
                raise ValueError(f"dim argument needed for 1D xp")
            else:
                interp = _interpolate_1d_constant_output_levels(xp, x, field, dim)
        else:
            interp = _interpolate_1d_variable_output_levels(xp, x, field)
        interp.attrs = field.attrs
        return interp


def _interpolate_1d_constant_output_levels(
    xp: xr.DataArray, x: xr.DataArray, field: xr.DataArray, dim: str
):

    output_grid = np.asarray(xp)
    out_dim = list(xp.dims)[0]

    def _interpolate(x: np.ndarray, field: np.ndarray) -> np.ndarray:
        # axis=-1 gives a broadcast error in the current version of metpy
        axis = field.ndim - 1
        return metpy.interpolate.interpolate_1d(output_grid, x, field, axis=axis)

    output = xr.apply_ufunc(
        _interpolate,
        x,
        field,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[out_dim]],
        output_sizes={out_dim: len(output_grid)},
        dask="parallelized",
        output_dtypes=[field.dtype],
    )

    # make the array have the same order of dimensions as before
    dim_order = [dim if dim in field.dims else out_dim for dim in output.dims]
    return output.transpose(*dim_order).assign_coords({out_dim: output_grid})


def _interpolate_2d(xp: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if mappm is not None:
        return mappm.interpolate_2d(xp, x, y, fill_value=np.nan)
    else:
        raise ModuleNotFoundError("mappm is not installed, required for this routine")


def _apply_2d(
    func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    xp: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    axis=0,
) -> np.ndarray:

    axis = axis % xp.ndim

    assert x.shape == y.shape

    sample_shape = tuple(size for n, size in enumerate(xp.shape) if n != axis)

    def flatten(x):
        swapped = x.swapaxes(axis, -1)
        return swapped.reshape((-1, swapped.shape[-1]))

    x_flat = flatten(x)
    y_flat = flatten(y)
    xp_flat = flatten(xp)
    output = func(xp_flat, x_flat, y_flat)
    reshaped = output.reshape((sample_shape + (-1,)))
    out = reshaped.swapaxes(-1, axis)
    return out


def _interpolate_1d_variable_output_levels(
    xp: xr.DataArray, x: xr.DataArray, y: xr.DataArray
) -> xr.DataArray:
    """Interpolate data along a single dimension

    Args:
        xp: the desired output coordinates
        x: the collocation points of the input data. must share all
            dimensions except 1 of the xp.
        y: the field to be interpolated. Must share dimensions with x.

    Returns:
        interpolated: field interpolated along the single dimension NOT
            shared by x and xp.

    """
    old_dim = (set(x.dims) - set(xp.dims)).pop()
    new_dim = (set(xp.dims) - set(x.dims)).pop()

    return xr.apply_ufunc(
        functools.partial(_apply_2d, _interpolate_2d, axis=-1),
        xp,
        x,
        y,
        input_core_dims=[[new_dim], [old_dim], [old_dim]],
        output_core_dims=[[new_dim]],
        dask="parallelized",
        output_sizes={new_dim: len(xp[new_dim])},
        output_dtypes=[y.dtype],
    )


def _coords_to_points(coords, order):
    return np.stack([coords[key] for key in order], axis=-1)


def interpolate_unstructured(
    data: Union[xr.DataArray, xr.Dataset], coords
) -> Union[xr.DataArray, xr.Dataset]:
    """Interpolate an unstructured dataset

    This is similar to the fancy indexing of xr.Dataset.interp, but it works
    with unstructured grids. Only nearest neighbors interpolation is supported for now.

    Args:
        data: data to interpolate
        coords: dictionary of dataarrays with single common dim, similar to the
            advanced indexing provided ``xr.DataArray.interp``. These can,
            but do not have to be actual coordinates of the Dataset, but they should
            be in a 1-to-1 map with the the dimensions of the data. For instance,
            one can use this function to find the height of an isotherm, provided
            that the temperature is monotonic with height.
    Returns:
        interpolated dataset with the coords from coords argument as coordinates.
    """
    dims_in_coords = set()
    for coord in coords:
        for dim in coords[coord].dims:
            dims_in_coords.add(dim)

    if len(dims_in_coords) != 1:
        raise ValueError(
            "The values of ``coords`` can only have one common shared "
            "dimension. The coords have these dimensions: "
            f"`{dims_in_coords}`"
        )

    dim_name = dims_in_coords.pop()

    spatial_dims = set()
    for key in coords:
        for dim in data[key].dims:
            spatial_dims.add(dim)

    stacked = data.stack({dim_name: list(spatial_dims)})
    order = list(coords)
    input_points = _coords_to_points(stacked, order)
    output_points = _coords_to_points(coords, order)
    tree = KDTree(input_points)
    _, indices = tree.query(output_points)
    output = stacked.isel({dim_name: indices})
    output = output.drop(dim_name)
    return output.assign_coords(coords)


def upsample_1d_periodic(arr: np.ndarray, upsample_factor):
    if len(arr.shape) > 1:
        raise ValueError("Array to upsample must be 1D.")
    x = 2 * np.pi * np.arange(0, arr.size) / (arr.size)
    x_upsampled = (
        2
        * np.pi
        * np.arange(0, upsample_factor * arr.size)
        / (upsample_factor * arr.size)
    )
    return np.interp(x_upsampled, x, arr, period=2 * np.pi)
