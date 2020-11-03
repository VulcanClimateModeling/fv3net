#!/bin/bash
set -e
set -x

[[ -f ./kustomize ]] || \
    ./examples/train-evaluate-prognostic-run/install_kustomize.sh 3.8.6

kustomizations="examples/train-evaluate-prognostic-run/"

for k in $kustomizations; do
    ./kustomize build $k
done
