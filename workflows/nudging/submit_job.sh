#!/bin/bash

set -e

usage="usage: submit_job.sh [-h] [-j JOB_PREFIX] \
       [-d] config runfile output_url docker_image pvc output_pvc \n\
       -j JOB_PREFIX: job name prefix for k8s submission \n\
       -d: detach job from terminal session" 
detach=0
job_prefix="nudge-to-highres"

while getopts "j:dh" OPTION; do
    case $OPTION in
        j)
            job_prefix=$OPTARG
        ;;
        h)
            echo -e $usage
            exit 1
        ;;
        d)
            detach=1
        ;;
        *)
            echo -e $usage
            exit 1
        ;;
    esac
done

shift $(($OPTIND - 1))

if [ $# -lt 5 ]; then
    echo -e $usage
    exit 1
fi

rand_tag=$(openssl rand --hex 4)

export CONFIG=$1; shift
export RUNFILE=runfile.py
export OUTPUT_URL=$1; shift
export DOCKER_IMAGE=$1; shift
export RESTARTS_PVC=$1; shift
export DYNAMIC_VOLUME=$1; shift

export JOBNAME=$job_prefix-$rand_tag
export NUDGING_CM=nudging-cm-$rand_tag


config_str=$(python prepare_config.py "$CONFIG")

cat <<EOF > dynamic_volume.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${DYNAMIC_VOLUME}
  labels:
    group: nudging-storage
spec:
  storageClassName: fast
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1.3Ti
EOF

kubectl create cm "$NUDGING_CM" --from-literal fv3config.yaml="$config_str" --from-file runfile.py="$RUNFILE"
kubectl apply -f dynamic_volume.yaml
envsubst < job_template.yaml | kubectl apply -f -

## JOB WAITING

SLEEP_TIME=120

function getJob {
    kubectl get job -n $1 $2 -o json
}

function waitForComplete {
    # Sleep while job is active
    NAMESPACE=$1
    JOBNAME=$2
    job_active=$(getJob $NAMESPACE $JOBNAME| jq --raw-output .status.active)
    echo -e "$job_active"
    while [[ $job_active == "1" ]]
    do
        echo -e "$(date '+%Y-%m-%d %H:%M')" Job active: "$JOBNAME" ... sleeping ${SLEEP_TIME}s
        sleep $SLEEP_TIME
        job_active=$(getJob $NAMESPACE $JOBNAME| jq --raw-output .status.active)
    done

    # Check for job success
    job_succeed=$(getJob $NAMESPACE $JOBNAME | jq --raw-output .status.succeeded)
    job_fail=$(getJob $NAMESPACE $JOBNAME | jq --raw-output .status.failed)
    if [[ $job_succeed == "1" ]]
    then
        echo -e Job successful: "$JOBNAME"
        kubectl delete pod $(kubectl get pod -l job-name=$JOBNAME -o json | jq --raw-output .items[].metadata.name)
        kubectl delete pvc ${DYNAMIC_VOLUME}
    elif [[ $job_fail == "1" ]]
    then
        echo -e Job failed: "$JOBNAME"
        exit 1
    else
        echo -e Job success ambiguous: "$JOBNAME"
        exit 1
    fi
}

if [[ $detach -ne 1 ]]; then
    waitForComplete default $JOBNAME
fi


