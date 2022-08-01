#!/bin/bash

set -xe

[[ -n $GOOGLE_APPLICATION_CREDENTIALS ]] && gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS

RUN=gs://vcm-ml-code-testing-data/sample-prognostic-run-output-v3

random=$(openssl rand --hex 6)
OUTPUT=gs://vcm-ml-scratch/test-prognostic-report/$random

cd workflows/diagnostics
# test shell
cat << EOF > report.script
load $RUN
print
hovmoller PWAT
avg2d PWAT
map2d PWAT
eval 3d.script
jupyter
EOF

cat << EOF > 3d.script
meridional specific_humidity
zonal specific_humidity
zonalavg specific_humidity
column specific_humidity
avg3d specific_humidity
EOF
prognostic_run_diags shell report.script
# assert an image has been output
[[ -f image.png ]]

# compute diagnostics/mterics for a short sample prognostic run
mkdir -p /tmp/$random
prognostic_run_diags save $RUN /tmp/$random/diags.nc --n-jobs=4
prognostic_run_diags metrics /tmp/$random/diags.nc > /tmp/$random/metrics.json
gsutil cp /tmp/$random/diags.nc $OUTPUT/run1/diags.nc
gsutil cp /tmp/$random/metrics.json $OUTPUT/run1/metrics.json

# generate movies for short sample prognostic run
prognostic_run_diags movie --n_jobs 1 --n_timesteps 2 $RUN $OUTPUT/run1

# generate report based on diagnostics computed above
prognostic_run_diags report $OUTPUT $OUTPUT

# cleanup
rm -r /tmp/$random

echo "Yay! Prognostic run report integration test passed. You can view the generated report at:"
echo "https://storage.cloud.google.com/vcm-ml-scratch/test-prognostic-report/${random}/index.html"

