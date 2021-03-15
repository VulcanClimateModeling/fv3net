    NUM_WORKERS=256


    python -m fv3net.pipelines.restarts_to_zarr  \
        --setup $(pwd)/setup.py \
        --job_name test-$(uuid) \
        --project vcm-ml \
        --region us-central1 \
        --runner DataFlow \
        --temp_location gs://vcm-ml-scratch/tmp_dataflow \
        --num_workers $NUM_WORKERS \
        --autoscaling_algorithm=NONE \
        --worker_machine_type n1-standard-1 \
        --disk_size_gb 30 \
        --url gs://vcm-ml-intermediate/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files \
        --output gs://vcm-ml-intermediate/2020-03-16-5-day-X-SHiELD-simulation-C384-restart-files.zarr
        # --n-steps 50  \