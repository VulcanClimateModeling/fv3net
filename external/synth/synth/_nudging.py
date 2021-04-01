import os
from typing import Sequence, Mapping

import numpy as np

from synth.core import DatasetSchema, generate, Range
import synth.core

from .schemas import load_schema_directory


def generate_nudging(outdir: str, times: Sequence[np.datetime64]):
    ranges = {"pressure_thickness_of_atmospheric_layer": Range(0.99, 1.01)}
    directory_schema = load_schema_directory("nudge_to_fine")
    synth.core.write_directory_schema(outdir, directory_schema, ranges)
