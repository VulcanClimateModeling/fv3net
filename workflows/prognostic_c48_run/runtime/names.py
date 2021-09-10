from typing import Mapping, Hashable

TEMP = "air_temperature"
TOTAL_WATER = "total_water"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
# [kg/m2/s], due to physics parmameterization
PHYSICS_PRECIP_RATE = "surface_precipitation_rate"
# [kg/m2/s], might also include nudging or ML contributions on top of physics
TOTAL_PRECIP_RATE = "total_precipitation_rate"
TOTAL_PRECIP = "total_precipitation"  # has units of m
AREA = "area_of_grid_cell"
EAST_WIND = "eastward_wind_after_physics"
NORTH_WIND = "northward_wind_after_physics"
SST = "ocean_surface_temperature"
TSFC = "surface_temperature"
MASK = "land_sea_mask"

# following variables are required no matter what feature set is being used
TENDENCY_TO_STATE_NAME: Mapping[Hashable, Hashable] = {
    "dQ1": TEMP,
    "dQ2": SPHUM,
    "dQu": EAST_WIND,
    "dQv": NORTH_WIND,
}
NUDGING_TENDENCY_SUFFIX = "tendency_due_to_nudging"
