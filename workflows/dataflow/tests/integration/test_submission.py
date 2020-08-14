import subprocess

submission = """
bash -x workflows/dataflow/dataflow.sh submit \
    workflows/dataflow/tests/integration/simple_dataflow.py  \
    --save_main_session \
    --job_name test-$(uuid) \
    --project vcm-ml \
    --region us-central1 \
    --runner DataFlow \
    --temp_location gs://vcm-ml-scratch/tmp_dataflow \
    --num_workers 1 \
    --autoscaling_algorithm=NONE \
    --worker_machine_type n1-standard-1 \
    --disk_size_gb 30
"""


def test_submit_dataflow():
    subprocess.check_call([submission], shell=True)
