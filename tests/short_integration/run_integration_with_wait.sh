#!/bin/bash

set -e

if [[ $# != 2 ]]
then
    echo "usage: tests/end_to_end_integration/run_integration_with_wait.sh <version> <integration_subfolder>"
    exit 1
fi

SLEEP_TIME=60

function getJob {
    kubectl get job -n $1 $2 -o json | jq '.items[0]'
}

function waitForComplete {
    # Sleep while job is active
    jobName=$1
    NAMESPACE=$2
    timeout=$(date -ud "50 minutes" +%s)
    job_active=$(getJob $NAMESPACE $jobName| jq --raw-output .status.active)
    echo "$job_active"
    while [[ $(date +%s) -le $timeout ]] && [[ $job_active == "1" ]]
    do
        echo "$(date '+%Y-%m-%d %H:%M')" Job active: "$jobName" ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_active=$(getJob $NAMESPACE $jobName| jq --raw-output .status.active)
    done

    # Check for job success
    job_succeed=$(getJob $NAMESPACE $jobName | jq --raw-output .status.succeeded)
    job_fail=$(getJob $NAMESPACE $jobName | jq --raw-output .status.failed)
    if [[ $job_succeed == "1" ]]
    then
        echo Job successful: "$jobName"
    elif [[ $job_fail == "1" ]]
    then
        echo Job failed: "$jobName"
        exit 1
    else
        echo Job timed out or success ambiguous: "$jobName"
        exit 1
    fi
}

VERSION=$1
SUBFOLDER=$2

cd tests/short_integration/$SUBFOLDER

random=$(openssl rand --hex 6)
suffix=-integration-test-$random
jobname=v1end-to-end${suffix}

(./kustomize_template.sh)

echo "Running tests with this kustomization.yaml:"
cat kustomization/kustomization.yaml

<<<<<<< HEAD
kubectl apply -k  kustomization --dry-run=client  -o yaml
kubectl apply -k kustomization
=======
kubectl apply -k  kustomization --dry-run  -o yaml
# kubectl apply -k kustomization
>>>>>>> parent of de35438f... Add test training only integration

trap "kubectl logs -lwaitForMe=\"$RANDOM_TAG\"" EXIT
waitForComplete -lwaitForMe="$RANDOM_TAG" default
