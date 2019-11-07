python pipeline.py  \
    --job_name convert-rundirs-to-zarr-$(uuidgen) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataflowRunner \
    --setup_file ./setup.py \
    --temp_location gs://vcm-ml-data/tmp_dataflow \
    --num_workers 64 \
    --max_num_workers 128 \
    --disk_size_gb 50 \
    --worker_machine_type n1-standard-1 
   # --service_account_email brian-vm-sa@vcm-ml.iam.gserviceaccount.com
