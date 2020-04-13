import pytest

from typing import Mapping

from fv3net.pipelines.kube_jobs import (
    wait_for_complete,
    get_base_fv3config,
    job_failed,
    job_complete,
)
from fv3net.pipelines.kube_jobs.utils import _handle_jobs
import kubernetes

from kubernetes.client import V1Job, V1JobStatus, V1ObjectMeta


failed_condition = kubernetes.client.V1JobCondition(type="Failed", status="True")

succesful_condition = kubernetes.client.V1JobCondition(type="Complete", status="True")

failed_status = V1JobStatus(active=None, conditions=[failed_condition])

complete_status = V1JobStatus(active=None, conditions=[succesful_condition])

inprogress_status = V1JobStatus(
    active=1, completion_time=None, conditions=None, failed=None, succeeded=None
)


def test_get_base_fv3config():

    config = get_base_fv3config("v0.3")
    assert isinstance(config, Mapping)


def test_get_base_fv3config_bad_version():

    with pytest.raises(KeyError):
        get_base_fv3config("nonexistent_fv3gfs_version_key")


@pytest.mark.parametrize(
    "func, status, expected",
    [
        (job_failed, complete_status, False),
        (job_failed, inprogress_status, False),
        (job_failed, failed_status, True),
        (job_complete, complete_status, True),
        (job_complete, inprogress_status, False),
        (job_complete, failed_status, False),
    ],
)
def test_job_failed(func, status, expected):
    job = V1Job(status=status)
    assert func(job) == expected


@pytest.mark.parametrize(
    "statuses, expected",
    [
        ([complete_status, complete_status, complete_status], True),
        ([complete_status, inprogress_status, complete_status], False),
        ([inprogress_status, inprogress_status, inprogress_status], False),
    ],
)
def test__handle_jobs_completed(statuses, expected):
    jobs = [
        V1Job(metadata=V1ObjectMeta(name=str(k)), status=status)
        for k, status in enumerate(statuses)
    ]
    assert _handle_jobs(jobs) == expected


def test__handle_jobs_raises_error():
    statuses = [complete_status, inprogress_status, failed_status]
    jobs = [
        V1Job(metadata=V1ObjectMeta(name=str(k)), status=status)
        for k, status in enumerate(statuses)
    ]

    with pytest.raises(ValueError):
        _handle_jobs(jobs)
