__version__ = "0.1.0"

from .config import VARNAMES
from .utils import (
    reduce_to_diagnostic,
    conditional_average,
    weighted_average,
    snap_mask_to_type,
)

__all__ = [
    "VARNAMES",
    "reduce_to_diagnostic",
    "conditional_average",
    "weighted_average",
    "snap_mask_to_type",
]
