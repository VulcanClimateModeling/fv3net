#!/bin/bash
set -x


if ! [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]]
then
    gcloud auth activate-service-account \
        --key-file "$GOOGLE_APPLICATION_CREDENTIALS"
fi

run=${1%/}  # strip possible trailing slash
output=$2
flags=$3

if [[ $run != gs://* ]]; then
    echo "memoized_compute_diagnostics.sh only works with Google Cloud Storage URL inputs. Got ${run}"
    exit 1
fi

cacheURL=${run}_diagnostics

# check for existence of diagnostics and metrics in cache
gsutil -q stat "$cacheURL/diags.nc"
diagsExitCode=$?
gsutil -q stat "$cacheURL/metrics.json"
metricsExitCode=$?

set -e

if [[ $diagsExitCode -eq 0 && $metricsExitCode -eq 0 ]]; then
    echo "Prognostic run diagnostics detected in cache for given run. Using cached diagnostics."
else
    echo "No prognostic run diagnostics detected in cache for given run. Computing diagnostics and adding to cache."	
    prognostic_run_diags save $flags "$run" diags.nc
    prognostic_run_diags metrics diags.nc > metrics.json
    gsutil cp diags.nc "$cacheURL/diags.nc"
    gsutil cp metrics.json "$cacheURL/metrics.json"
fi

gsutil cp "$cacheURL/diags.nc" "$output/diags.nc"
gsutil cp "$cacheURL/metrics.json" "$output/metrics.json"
