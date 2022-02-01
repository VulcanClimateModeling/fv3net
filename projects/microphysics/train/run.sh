#!/bin/bash

set -e
set -o pipefail

if [[ "$1" == "--test" ]]; then
    extra_flags="--nfiles 2 --nfiles_valid 2 --epochs 5"
    bucket="vcm-ml-scratch"
else
    bucket="vcm-ml-experiments"
fi

group="$(openssl rand -hex 3)"
config=rnn
config_file="${config}.yaml"

for lr in 0.0001 0.001 0.01
do
    model_name="rnn-alltdep-${group}-lr${lr}"
    out_url=$(artifacts resolve-url "$bucket" microphysics-emulation "${model_name}")
    argo submit argo.yaml \
        --name "${model_name}" \
        -p training-config="$(base64 --wrap 0 $config_file)" \
        -p flags="--out_url ${out_url} ${extra_flags} --loss.optimizer.kwargs.learning_rate $lr" | tee -a submitted-jobs.txt
done
