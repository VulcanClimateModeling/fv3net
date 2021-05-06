#!/bin/bash -e

set -e

# TODO need to generate sdists with setup.py files with no requirements

DATAFLOW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROOT=$DATAFLOW_ROOT/../..

vcmPackages=(
  "${ROOT}/workflows/dataflow"
  "${ROOT}/external/vcm"
  "${ROOT}/workflows/fine_res_budget"
  "${ROOT}/external/xpartition"
  "${ROOT}/external/gcsfs"
)

function buildSdist {
  (
    cd "$1"
    rm -rf dist
    python setup.py sdist > /dev/null 2> /dev/null
    cp dist/*.tar.gz "$2"
  )
}

function buildPackages {
  rm -rf "$1"
  mkdir -p "$1"
  for package in "${vcmPackages[@]}"
  do
    >&2 echo "Building $package"
    buildSdist "$package" "$1"
  done
}

function checkInstallation {
  cd "$(prepareWorkingDirectory)"
  python -m venv env
  source env/bin/activate

  pip install apache_beam
  pip install dists/vcm*.tar.gz
  pip install dists/budget*.tar.gz
  pip install dists/xpartition*.tar.gz
  pip install dists/gcsfs*.tar.gz
}

function runRemote {
  cd "$(prepareWorkingDirectory)"

  packageArgs=" \
  --extra_package dists/vcm*.tar.gz \
  --extra_package dists/fv3net*.tar.gz \
  --extra_package dists/budget*.tar.gz \
  --extra_package dists/xpartition*.tar.gz \
  --extra_package dists/gcsfs*.tar.gz \
  "
  
  cmd="python $* $packageArgs"
  echo "Running: $cmd"
  $cmd
}

function usage {
  echo "Submit a dataflow job"
  echo ""
  echo "Usage:"
  echo "  dataflow.sh submit (-m <module> | <absolute_path>) <args>..."
  echo "  dataflow.sh check"
  echo "  dataflow.sh -h"
  echo ""
  echo "Commands:"
  echo "" 
  echo "  submit     submit a remote dataflow job" 
  echo "  check      recreate the dataflow environment setup in a local virtualenv" 
  echo ""
  echo "Options:"
  echo "  -h         Show the help"
}

function prepareWorkingDirectory {
  workdir=$(mktemp -d)
  buildPackages "$workdir/dists/"
  echo "$workdir"
}

if [[ $# -lt 1 ]]
then
  usage
  exit 2
fi


subcommand="$1"
shift

case $subcommand in 
  submit)
    runRemote "$@"
    ;;
  check)
    checkInstallation
    ;;
  -h)
    usage
    exit 2
    ;;
  *)
    >&2 echo "invalid subcommand: $subcommand"
    exit 2
    ;;
esac
