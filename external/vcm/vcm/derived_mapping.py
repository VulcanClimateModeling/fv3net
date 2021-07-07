import numpy as np
from typing import Mapping, Hashable, Callable, Iterable
import xarray as xr

import vcm


class DerivedMapping:
    """A uniform mapping-like interface for both existing and derived variables.
    
    Allows register and computing derived variables transparently in either
    the FV3GFS state or a saved dataset.

    """

    _VARIABLES: Mapping[Hashable, Callable[..., xr.DataArray]] = {}

    def __init__(self, mapper: Mapping[Hashable, xr.DataArray]):
        self._mapper = mapper

    @classmethod
    def register(cls, name: Hashable):
        """Register a function as a derived variable

        Args:
            name: the name the derived variable will be available under
        """

        def decorator(func):
            cls._VARIABLES[name] = func
            return func

        return decorator

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key in self._VARIABLES:
            return self._VARIABLES[key](self)
        else:
            return self._mapper[key]

    def keys(self):
        return set(self._mapper) | set(self._VARIABLES)

    def _data_arrays(self, keys: Iterable[Hashable]):
        return {key: self[key] for key in keys}

    def dataset(self, keys: Iterable[Hashable]) -> xr.Dataset:
        return xr.Dataset(self._data_arrays(keys))


@DerivedMapping.register("cos_zenith_angle")
def cos_zenith_angle(self):
    return vcm.cos_zenith_angle(self["time"], self["lon"], self["lat"])


@DerivedMapping.register("evaporation")
def evaporation(self):
    lhf = self["latent_heat_flux"]
    return vcm.thermo.latent_heat_flux_to_evaporation(lhf)


def _rotate(self: DerivedMapping, x, y):
    wind_rotation_matrix = self.dataset(
        [
            "eastward_wind_u_coeff",
            "eastward_wind_v_coeff",
            "northward_wind_u_coeff",
            "northward_wind_v_coeff",
        ]
    )
    return vcm.cubedsphere.center_and_rotate_xy_winds(
        wind_rotation_matrix, self[x], self[y]
    )


@DerivedMapping.register("dQu")
def dQu(self):
    try:
        return self._mapper["dQu"]
    except (KeyError):
        return _rotate(self, "dQxwind", "dQywind")[0]


@DerivedMapping.register("dQv")
def dQv(self):
    try:
        return self._mapper["dQv"]
    except (KeyError):
        return _rotate(self, "dQxwind", "dQywind")[1]


@DerivedMapping.register("eastward_wind")
def eastward_wind(self):
    try:
        return self._mapper["eastward_wind"]
    except (KeyError):
        return _rotate(self, "x_wind", "y_wind")[0]


@DerivedMapping.register("northward_wind")
def northward_wind(self):
    try:
        return self._mapper["northward_wind"]
    except (KeyError):
        return _rotate(self, "x_wind", "y_wind")[1]


@DerivedMapping.register("dQu_parallel_to_eastward_wind")
def dQu_parallel_to_eastward_wind_direction(self):
    sign = np.sign(self["eastward_wind"] / self["dQu"])
    return sign * abs(self["dQu"])


@DerivedMapping.register("dQv_parallel_to_northward_wind")
def dQv_parallel_to_northward_wind_direction(self):
    sign = np.sign(self["northward_wind"] / self["dQv"])
    return sign * abs(self["dQv"])


@DerivedMapping.register("horizontal_wind_tendency_parallel_to_horizontal_wind")
def horizontal_wind_tendency_parallel_to_horizontal_wind(self):
    tendency_projection_onto_wind = (
        self["eastward_wind"] * self["dQu"] + self["northward_wind"] * self["dQv"]
    ) / np.linalg.norm((self["eastward_wind"], self["northward_wind"]))
    return tendency_projection_onto_wind


@DerivedMapping.register("net_shortwave_sfc_flux_derived")
def net_shortwave_sfc_flux_derived(self):
    # Positive = downward direction
    albedo = self["surface_diffused_shortwave_albedo"]
    downward_sfc_shortwave_flux = self[
        "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface"
    ]
    return (1 - albedo) * downward_sfc_shortwave_flux
