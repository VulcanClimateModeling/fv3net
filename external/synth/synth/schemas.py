import os

from synth.core import DatasetSchema, load, load_directory_schema

CURRENT_DIR = os.path.dirname(__file__)


def load_schema(name: str) -> DatasetSchema:
    """Load a schema by name from the current library"""
    path = os.path.join(CURRENT_DIR, "_dataset_fixtures", name)
    with open(path) as f:
        return load(f)


def load_schema_directory(name: str) -> DatasetSchema:
    """Load a schema by name from the current library"""
    path = os.path.join(CURRENT_DIR, "_dataset_fixtures", name)
    return load_directory_schema(path)
