import argparse
from . import core


def read_schema():
    parser = argparse.ArgumentParser(
        description="Save schema for a directory of zarrs to another directory"
    )
    parser.add_argument("src")
    parser.add_argument("dest")
    args = parser.parse_args()

    schema = core.read_directory_schema(args.src)
    core.dump_directory_schema_to_disk(schema, args.dest)
