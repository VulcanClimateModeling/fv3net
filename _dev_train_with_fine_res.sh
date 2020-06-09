#!/bin/bash

cat <<EOF >config.yaml
model_type: sklearn_random_forest
hyperparameters:
  max_depth: 4
  n_estimators: 2
mapping_function: open_one_step
input_variables:
  - air_temperature
  - specific_humidity
output_variables:
  - dQ1
  - dQ2
batch_kwargs:
  num_batches: 2
  timesteps_per_batch: 1
  init_time_dim_name: "initial_time"
EOF

python -m fv3net.regression.sklearn -h
