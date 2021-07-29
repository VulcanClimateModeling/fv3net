from typing import Tuple

import xarray as xr
from .coarsen import shift_edge_var_to_center

EDGE_TO_CENTER_DIMS = {"x_interface": "x", "y_interface": "y"}


def center_and_rotate_xy_winds(
    wind_rotation_matrix: xr.Dataset,
    x_component: xr.DataArray,
    y_component: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """ Transform D grid x/y winds to A grid E/N winds.

    Args:
        wind_rotation_matrix : Dataset with rotation coefficients for
        x/y to E/N rotation. Can be found in catalog.
        x_component : D grid x wind
        y_component : D grid y wind

    Returns:
        Tuple of eastward and northward winds on A-grid.
    """
    common_coords = {
        "x": wind_rotation_matrix["x"].values,
        "y": wind_rotation_matrix["y"].values,
    }
    x_component_centered = shift_edge_var_to_center(
        x_component, EDGE_TO_CENTER_DIMS
    ).assign_coords(common_coords)
    y_component_centered = shift_edge_var_to_center(
        y_component, EDGE_TO_CENTER_DIMS
    ).assign_coords(common_coords)
    return rotate_xy_winds(
        wind_rotation_matrix, x_component_centered, y_component_centered
    )


def rotate_xy_winds(
    wind_rotation_matrix: xr.Dataset,
    x_wind_centered: xr.DataArray,
    y_wind_centered: xr.DataArray,
) -> Tuple[xr.DataArray, xr.DataArray]:
    eastward = (
        wind_rotation_matrix["eastward_wind_u_coeff"] * x_wind_centered
        + wind_rotation_matrix["eastward_wind_v_coeff"] * y_wind_centered
    )
    northward = (
        wind_rotation_matrix["northward_wind_u_coeff"] * x_wind_centered
        + wind_rotation_matrix["northward_wind_v_coeff"] * y_wind_centered
    )
    return (
        eastward.transpose(*x_wind_centered.dims),
        northward.transpose(*y_wind_centered.dims),
    )
