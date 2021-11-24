#!/bin/bash

set -e

echo "$1" | base64 --decode > template.yaml
shift
MONTH=$1
shift
TAG=$1
shift
OUTPUT_FREQUENCY=$1

# add initial condition to template
TIMESTAMP=$(printf "2016%02d0100" $MONTH)

# to replace initial_conditions in template
export IC_URL="gs://vcm-ml-raw/2020-11-05-GFS-month-start-initial-conditions-year-2016/$TIMESTAMP"
export MONTH_INT=$(printf '%d' $MONTH)
envsubst < template.yaml > fv3config.yaml

echo "Running the following configuration"
cat fv3config.yaml

python3 ../scripts/prognostic_run.py \
    --tag "$TAG" \
    --model NO_MODEL \
    --config-path fv3config.yaml \
    --output-frequency "$OUTPUT_FREQUENCY" \
    --offline