#!/bin/bash

#python evaluate_prescribed_run.py \
#    gs://vcm-ml-experiments/default/2021-11-22/n2f-prescribe-apparent-sources-only/fv3gfs_run \
#    gs://vcm-ml-public/oliwm/compare-fine-res-corrections/apparent-sources-only

#python evaluate_prescribed_run.py \
#    gs://vcm-ml-experiments/default/2021-11-22/n2f-prescribe-apparent-sources-extend-lower/fv3gfs_run \
#    gs://vcm-ml-public/oliwm/compare-fine-res-corrections/apparent-sources-extend-lower

#python evaluate_prescribed_run.py \
#    gs://vcm-ml-experiments/default/2021-11-22/n2f-prescribe-apparent-sources-plus-nudging/fv3gfs_run \
#    gs://vcm-ml-public/oliwm/compare-fine-res-corrections/apparent-sources-plus-nudging

python evaluate_prescribed_run.py \
    gs://vcm-ml-experiments/default/2021-12-09/n2f-prescribe-dynamics-difference/fv3gfs_run \
    gs://vcm-ml-public/oliwm/compare-fine-res-corrections/dynamics-difference

