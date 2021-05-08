import pkg_resources
import re
import subprocess
import tempfile
import sys


def clean_version(version):
    return re.match(r"[0-9\.]*", version).group(0).rstrip(".")


def dump_versions(packages, f):
    for package, version in packages.items():
        print(f"{package}=={clean_version(version)}", file=f)


def load_versions(f):
    versions = {}
    for line in f:
        if "==" in line:
            package, version_with_extra = line.split("==")
            versions[package] = clean_version(version_with_extra)
    return versions


def _compile(requirements, existing_versions):
    with tempfile.NamedTemporaryFile("w") as f:
        dump_versions(existing_versions, f)
        subprocess.check_call(
            ["pip-compile", "--output-file", f.name] + requirements,
            stderr=subprocess.DEVNULL,
        )
        with open(f.name, "r") as f_read:
            return load_versions(f_read)


def version_diff(old, new):
    updates = {}
    for package, version in new.items():
        if package not in old:
            print(f"Adding {package} at {version}", file=sys.stderr)
            updates[package] = version
        elif old[package] != version:
            old_version = old[package]
            print(
                f"Updating {package} from {old_version} to {version}", file=sys.stderr
            )
            updates[package] = version
    return updates


def list_packages():
    existing_versions = {}
    for pkg in pkg_resources.working_set:
        existing_versions[pkg.project_name] = pkg.version
    return existing_versions


assert clean_version("0.1.1") == "0.1.1"
assert clean_version("0.12.2.dev") == "0.12.2"


requirements_files = sys.argv[1:]

existing_versions = list_packages()
compiled_versions = _compile(requirements_files, existing_versions)
updates = version_diff(existing_versions, compiled_versions)
dump_versions(updates, sys.stdout)
