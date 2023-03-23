import xarray as xr
from typing import Optional, List
from dataclasses import dataclass, field
import numpy as np

PRECIP_RATE = "total_precip_to_surface"

WVP = "water_vapor_path"
COL_DRYING = "minus_column_integrated_q2"
COL_MOISTENING = "column_integrated_q2"
HISTOGRAM_BINS = {
    PRECIP_RATE: np.logspace(-1, np.log10(500), 101),
    WVP: np.linspace(-10, 90, 101),
    COL_DRYING: np.linspace(-50, 150, 101),
    COL_MOISTENING: np.linspace(-150, 50, 101),
}

# argument typehint for diags in save_prognostic_run_diags but used
# by multiple modules split out to group operations and simplify
# the diagnostic script


@dataclass
class DiagArg:
    prediction: xr.Dataset
    verification: xr.Dataset
    grid: xr.Dataset
    delp: Optional[xr.DataArray] = None
    horizontal_dims: List[str] = field(default_factory=["x", "y", "tile"])
    vertical_dim: str = "z"
