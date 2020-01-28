import numpy as np
import xarray as xr

from vcm.cubedsphere.constants import INIT_TIME_DIM, FORECAST_TIME_DIM, COORD_Z_CENTER

gravity = 9.81
specific_heat = 1004


def mass_integrate(phi, dp, dim=COORD_Z_CENTER):
    return (phi * dp / gravity).sum(dim)


def apparent_heating(dtemp_dt, w):
    return dtemp_dt + w * gravity / specific_heat


def timedelta_to_seconds(dt):
    one_second = np.timedelta64(1000000000, "ns")
    return dt / one_second


def apparent_source(
    q: xr.DataArray, t_dim: str = INIT_TIME_DIM, s_dim: str = FORECAST_TIME_DIM
) -> xr.DataArray:
    """Compute the apparent source from stepped output

    Args:
        q: The variable to compute the source of
        t_dim, optional: the dimension corresponding to the initial condition
        s_dim, optional: the dimension corresponding to the forecast time

    Returns:
        The apparent source of q. Has units [q]/s

    """

    t = q[t_dim]
    s = q[s_dim]
    q = q.drop([t_dim, s_dim])
    dq = q.diff(t_dim)
    dq_c48 = q.diff(s_dim)
    dt = timedelta_to_seconds(t.diff(t_dim))
    ds = timedelta_to_seconds(s.diff(s_dim))

    tend = dq / dt
    tend_c48 = dq_c48 / ds

    # restore coords
    tend = tend.isel({s_dim: 0}).assign_coords(**{t_dim: t[:-1]})
    tend_c48 = tend_c48.isel({s_dim: 0, t_dim: slice(0, -1)}).assign_coords(
        **{t_dim: t[:-1]}
    )

    return tend - tend_c48
