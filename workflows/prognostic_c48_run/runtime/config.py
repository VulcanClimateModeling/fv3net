import os
from typing import List, Optional, Union
import dataclasses
import yaml
import f90nml

import dacite

from runtime.diagnostics.manager import (
    DiagnosticFileConfig,
    FortranFileConfig,
    get_chunks,
)
from runtime.steppers.nudging import NudgingConfig
from runtime.steppers.machine_learning import MachineLearningConfig
from runtime.steppers.prescriber import PrescriberConfig

FV3CONFIG_FILENAME = "fv3config.yml"


@dataclasses.dataclass
class UserConfig:
    """The top-level object for python runtime configurations

    Attributes:
        diagnostics: list of diagnostic file configurations
        fortran_diagnostics: list of Fortran diagnostic outputs. Currently only used by
            post-processing and so only name and chunks items need to be specified.
        prephysics: optional configuration of computations prior to physics,
            specified by either a machine learning configuation or a prescriber
            configuration
        scikit_learn: a machine learning configuration
        nudging: nudge2fine configuration. Cannot be used if any scikit_learn model
            urls are specified.
        step_tendency_variables: variables to compute the tendencies of.
            These could in principle be inferred from the requested diagnostic
            names.
        step_storage_variables: variables to compute the storage of. Needed for certain
            diagnostics.
    """

    diagnostics: List[DiagnosticFileConfig] = dataclasses.field(default_factory=list)
    fortran_diagnostics: List[FortranFileConfig] = dataclasses.field(
        default_factory=list
    )
    prephysics: Optional[Union[PrescriberConfig, MachineLearningConfig]] = None
    scikit_learn: Optional[MachineLearningConfig] = None
    nudging: Optional[NudgingConfig] = None
    step_tendency_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(
            ("specific_humidity", "air_temperature", "eastward_wind", "northward_wind",)
        )
    )
    step_storage_variables: List[str] = dataclasses.field(
        default_factory=lambda: list(("specific_humidity", "total_water"))
    )


def get_config() -> UserConfig:
    """Open the configurations for this run
    
    .. warning::
        Only valid at runtime
    """
    with open("fv3config.yml") as f:
        config = yaml.safe_load(f)
    return dacite.from_dict(UserConfig, config)


def get_namelist() -> f90nml.Namelist:
    """Open the fv3 namelist
    
    .. warning::
        Only valid at runtime
    """
    return f90nml.read("input.nml")


def write_chunks(config: UserConfig):
    """Given UserConfig, write chunks to 'chunks.yaml'"""
    diagnostic_file_configs = (
        config.fortran_diagnostics + config.diagnostics  # type: ignore
    )
    chunks = get_chunks(diagnostic_file_configs)
    with open("chunks.yaml", "w") as f:
        yaml.safe_dump(chunks, f)


def get_existing_rundir_items(ignore: Sequence[str]) -> Sequence[str]:
    """Return list of files that exist in rundir except those listed in ignore.
    
    .. warning::
        Only valid at runtime
    """
    items = []
    for root, dirs, files in os.walk("."):
        for name in files:
            items.append(os.path.join(root, name))
    items = [os.path.relpath(item, ".") for item in items]
    for item in ignore:
        if item in items:
            items.remove(item)
    return items


def write_existing_rundir_items(
    filename: str = "existing_files.yaml",
    ignore: Sequence[str] = ("time_stamp.out", "logs.txt", "fv3config.yml"),
):
    """Write list of files which currently exist in rundir except those listed in 'ignore'.
    
    .. warning::
        Only valid at runtime
    """
    items = get_existing_rundir_items(ignore)
    with open(filename, "w") as f:
        yaml.safe_dump(items, f)
