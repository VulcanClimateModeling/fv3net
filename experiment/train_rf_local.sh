
cat <<EOF >config.yaml
model_type: sklearn_random_forest
hyperparameters:
  max_depth: 13
  n_estimators: 1
input_variables:
- air_temperature
- specific_humidity
output_variables:
- dQ1
- dQ2
batch_function: batches_from_geodata
batch_kwargs:
  timesteps_per_batch: 10
  mapping_function: open_fine_res_apparent_sources
  mapping_kwargs:
    offset_seconds: 450
EOF

cat << EOF > times.json
[
  "20160817.101500",
  "20160807.194500",
  "20160806.023000",
  "20160815.131500",
  "20160827.154500",
  "20160820.073000",
  "20160811.073000",
  "20160822.090000",
  "20160831.114500",
  "20160831.081500",
  "20160820.133000",
  "20160811.034500",
  "20160816.040000",
  "20160824.041500",
  "20160825.210000",
  "20160805.011500",
  "20160822.020000",
  "20160826.073000",
  "20160827.030000",
  "20160813.101500",
  "20160809.041500",
  "20160813.130000",
  "20160831.003000",
  "20160806.193000",
  "20160806.071500",
  "20160821.130000",
  "20160829.040000",
  "20160807.180000",
  "20160819.184500",
  "20160813.091500",
  "20160821.050000",
  "20160815.234500",
  "20160806.144500",
  "20160815.020000",
  "20160822.113000",
  "20160829.124500",
  "20160813.021500",
  "20160828.014500",
  "20160822.233000",
  "20160819.201500",
  "20160814.003000",
  "20160814.151500",
  "20160816.041500",
  "20160831.111500",
  "20160808.013000",
  "20160824.180000",
  "20160826.190000",
  "20160807.054500",
  "20160805.084500",
  "20160831.191500",
  "20160823.040000",
  "20160813.081500",
  "20160829.060000",
  "20160826.223000",
  "20160818.103000",
  "20160808.184500",
  "20160810.021500",
  "20160828.030000",
  "20160820.140000",
  "20160815.100000",
  "20160817.210000",
  "20160828.071500",
  "20160824.094500",
  "20160809.094500",
  "20160824.231500",
  "20160823.123000",
  "20160825.024500",
  "20160812.210000",
  "20160816.140000",
  "20160827.214500",
  "20160829.231500",
  "20160829.193000",
  "20160806.181500",
  "20160805.023000",
  "20160819.003000",
  "20160814.170000",
  "20160810.204500",
  "20160807.164500",
  "20160807.151500",
  "20160813.160000",
  "20160829.073000",
  "20160814.053000",
  "20160824.093000",
  "20160815.024500",
  "20160814.094500",
  "20160823.091500",
  "20160821.053000",
  "20160809.150000",
  "20160821.004500",
  "20160823.033000",
  "20160809.131500",
  "20160821.220000",
  "20160823.014500",
  "20160831.101500",
  "20160831.233000",
  "20160818.134500",
  "20160829.174500",
  "20160806.170000",
  "20160811.090000",
  "20160810.210000",
  "20160822.194500",
  "20160805.033000",
  "20160829.213000",
  "20160825.113000",
  "20160810.201500",
  "20160829.070000",
  "20160825.070000",
  "20160828.060000",
  "20160820.090000",
  "20160818.234500",
  "20160818.064500",
  "20160809.173000",
  "20160830.054500",
  "20160827.120000",
  "20160819.083000",
  "20160825.090000",
  "20160821.173000",
  "20160823.080000",
  "20160811.134500",
  "20160826.203000",
  "20160823.223000",
  "20160818.164500",
  "20160830.061500",
  "20160809.083000",
  "20160815.174500",
  "20160819.020000",
  "20160805.051500",
  "20160812.121500",
  "20160828.194500",
  "20160814.000000"
]
EOF

python -m fv3fit.sklearn \
  gs://vcm-ml-experiments/2020-07-30-fine-res \
  config.yaml \
  gs://vcm-ml-scratch/noah/sklearn/$EXPERIMENT \
  --timesteps-file times.json \
  --no-train-subdir-append

