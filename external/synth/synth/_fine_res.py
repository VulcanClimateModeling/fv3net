import os
from typing import Sequence

from synth.core import write_directory_schema
from synth.schemas import load_schema_directory


def generate_fine_res(datadir: str):
    """Generate a directory of fine-res data
    
    Args:
        datadir: output location
        times: list of YYYYMMDD.HHMMSS timestamps
    
    """
    schema = load_schema_directory("fine_res_budget")
    write_directory_schema(datadir, schema)
