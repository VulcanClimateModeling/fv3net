#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 10 --nfiles_valid 10 --epochs 2"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

tag="9a68fb66d235223056812cc302fe0c2fe2717a53"
group="$(openssl rand -hex 3)"


for exp in limiter-all-loss limiter-tendency-loss no-limiter-tendency-loss
do
for arch in dense rnn
do
for ant in true false
do
    config_file="${exp}/${arch}-alltdep.yaml"
    model_name="${exp}-${arch}-alltdep-antarc-${ant}-${group}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    flags="--out_url ${out_url} ${extra_flags} --transform.antarctic_only ${ant}"
    argo submit ../train/argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="$flags" \
        -p wandb-run-group="antarctic-experiments-v2-feb-2022" \
        -p tag="${tag}"
done
done
done