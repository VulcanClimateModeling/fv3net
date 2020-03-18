import numpy as np
import xarray as xr
from ..cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER


# following are defined as in FV3GFS model (see FV3/fms/constants/constants.f90)
_GRAVITY = 9.80665  # m /s2
_RDGAS = 287.05  # J / K / kg
_RVGAS = 461.5  # J / K / kg
_LATENT_HEAT_VAPORIZATION_0_C = 2.5e6
_SPECIFIC_ENTHALPY_LIQUID = 4185.5
_SPECIFIC_ENTHALPY_VAP0R = 1846
_SPECIFIC_HEAT_CONST_PRESSURE = 1004
_FREEZING_TEMPERATURE = 273.15
_POISSON_CONST = 0.2854

_DEFAULT_SURFACE_TEMPERATURE = _FREEZING_TEMPERATURE + 15

_TOA_PRESSURE = 300.0  # Pa
_REFERENCE_SURFACE_PRESSURE = 100000  # reference pressure for potential temp [Pa]
_REVERSE = slice(None, None, -1)

_SEC_PER_DAY = 86400



def potential_temperature(P, T):
    return T * (_REFERENCE_SURFACE_PRESSURE / P) ** _POISSON_CONST


def pressure_at_interface(
    delp, toa_pressure=_TOA_PRESSURE, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER
):
    """ Compute pressure at layer interfaces

    Args:
        delp (xr.DataArray): pressure thicknesses
        toa_pressure (float, optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim_center (str, optional): name of vertical dimension for delp.
            Defaults to "pfull".
        dim_outer (str, optional): name of vertical dimension for output pressure.
            Defaults to "phalf".

    Returns:
        xr.DataArray: atmospheric pressure at layer interfaces
    """
    top = xr.full_like(delp.isel({dim_center: [0]}), toa_pressure).variable
    delp_with_top = top.concat([top, delp.variable], dim=dim_center)
    pressure = delp_with_top.cumsum(dim_center)
    return _add_coords_to_interface_variable(
        pressure, delp, dim_center=dim_center
    ).rename({dim_center: dim_outer})


def height_at_interface(dz, phis, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    """ Compute geopotential height at layer interfaces

    Args:
        dz (xr.DataArray): layer height thicknesses
        phis (xr.DataArray): surface geopotential
        dim_center (str, optional): name of vertical dimension for dz.
            Defaults to "pfull".
        dim_outer (str, optional): name of vertical dimension for output heights.
            Defaults to "phalf".

    Returns:
        xr.DataArray: height at layer interfaces
    """
    bottom = phis.broadcast_like(dz.isel({dim_center: [0]})) / _GRAVITY
    dzv = -dz.variable  # dz is negative in model
    dz_with_bottom = dzv.concat([dzv, bottom], dim=dim_center)
    height = (
        dz_with_bottom.isel({dim_center: _REVERSE})
        .cumsum(dim_center)
        .isel({dim_center: _REVERSE})
    )
    return _add_coords_to_interface_variable(height, dz, dim_center=dim_center).rename(
        {dim_center: dim_outer}
    )


def _add_coords_to_interface_variable(
    dv_outer: xr.Variable, da_center: xr.DataArray, dim_center: str = COORD_Z_CENTER
):
    """Assign all coords except vertical from da_center to dv_outer """
    if dim_center in da_center.coords:
        return xr.DataArray(dv_outer, coords=da_center.drop(dim_center).coords)
    else:
        return xr.DataArray(dv_outer, coords=da_center.coords)


def pressure_at_midpoint(delp, toa_pressure=_TOA_PRESSURE, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints by linear interpolation

    Args:
        delp (xr.DataArray): pressure thicknesses
        toa_pressure (float, optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (str, optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        xr.DataArray: atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(delp, toa_pressure=toa_pressure, dim_center=dim)
    return _interface_to_midpoint(pi, dim_center=dim)


def height_at_midpoint(dz, phis, dim=COORD_Z_CENTER):
    """ Compute geopotential height at layer midpoints by linear interpolation

    Args:
        dz (xr.DataArray): layer height thicknesses
        phis (xr.DataArray): surface geopotential
        dim (str, optional): name of vertical dimension for dz. Defaults to "pfull".

    Returns:
        xr.DataArray: height at layer midpoints
    """
    zi = height_at_interface(dz, phis, dim_center=dim)
    return _interface_to_midpoint(zi, dim_center=dim)


def _interface_to_midpoint(da, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    da_mid = (
        da.isel({dim_outer: slice(0, -1)}) + da.isel({dim_outer: slice(1, None)})
    ) / 2
    return da_mid.rename({dim_outer: dim_center})


def pressure_at_midpoint_log(delp, toa_pressure=_TOA_PRESSURE, dim=COORD_Z_CENTER):
    """ Compute pressure at layer midpoints following Eq. 3.17 of Simmons
    and Burridge (1981), MWR.

    Args:
        delp (xr.DataArray): pressure thicknesses
        toa_pressure (float, optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (str, optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        xr.DataArray: atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(
        delp, toa_pressure=toa_pressure, dim_center=dim, dim_outer=dim
    )
    dlogp = np.log(pi).diff(dim)
    return delp / dlogp


def hydrostatic_dz(T, q, delp, dim=COORD_Z_CENTER):
    """ Compute layer thickness assuming hydrostatic balance

    Args:
        T (xr.DataArray): temperature
        q (xr.DataArray): specific humidity
        delp (xr.DataArray): pressure thickness
        dim (str, optional): name of vertical dimension. Defaults to "pfull".

    Returns:
        xr.DataArray: layer thicknesses dz
    """
    pi = pressure_at_interface(delp, dim_center=dim, dim_outer=dim)
    tv = T * (1 + (_RVGAS / _RDGAS - 1) * q)
    dlogp = np.log(pi).diff(dim)
    return -dlogp * _RDGAS * tv / _GRAVITY


def dz_and_top_to_phis(top_height, dz, dim=COORD_Z_CENTER):
    """ Compute surface geopotential from model top height and layer thicknesses"""
    return _GRAVITY * (top_height + dz.sum(dim=dim))


def latent_heat_vaporization(T):
    return _LATENT_HEAT_VAPORIZATION_0_C + (
        _SPECIFIC_ENTHALPY_LIQUID - _SPECIFIC_ENTHALPY_VAP0R
    ) * (T - _FREEZING_TEMPERATURE)


def net_heating(
    dlw_sfc,
    dsw_sfc,
    ulw_sfc,
    ulw_toa,
    usw_sfc,
    usw_toa,
    dsw_toa,
    shf,
    surface_rain_rate,
    surface_temperature=_FREEZING_TEMPERATURE + 10,
):
    """A dataarray implementation of ``net_heating_from_dataset``

    All argument except for ``cond_int`` should be in W/m2. cond_int is in units
    kg/kg/s which is equivalent to a surface precipitation rate in mm/s. All
    precipitation is assumed to fall as liquid water.

    If provided, the surface temperature in K will be used to calculate the latent heat
    of vaporization with the FV3 formulas.
    """

    lv = latent_heat_vaporization(surface_temperature)
    da = (
        -dlw_sfc
        - dsw_sfc
        + ulw_sfc
        - ulw_toa
        + usw_sfc
        - usw_toa
        + dsw_toa
        + shf
        + surface_rain_rate * lv
    )
    da.attrs = {
        "long_name": "net heating from model physics",
        "units": "W/m^2"}
    return da


def latent_heat_flux_to_evaporation(
    lhf, surface_temperature=_DEFAULT_SURFACE_TEMPERATURE
):
    """Compute evaporation from latent heat flux

    Args:
        lhf: W/m2
        surface_temperature: degrees K

    Returns:
        evaporation: kg/s/m^2
    """

    return lhf / latent_heat_vaporization(surface_temperature)


def net_precipitation(lhf, prate):
    da = (prate - latent_heat_flux_to_evaporation(lhf)) * _SEC_PER_DAY
    da.attrs = {
        "long_name": "net precipitation from model physics",
        "units": "mm/day",
    },
    return da
