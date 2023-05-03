import io
import os
import sys
import tempfile
import warnings

import fv3config
import vcm
from fv3net.artifacts.metadata import StepMetadata

from fv3post.append import append_segment
from fv3post.post_process import post_process
from runtime.config import get_model_urls
from .run import run_segment


def read_last_segment(run_url):
    fs = vcm.get_fs(run_url)
    artifacts_dir = os.path.join(run_url, "artifacts")
    # fsspec caches the ls call locally so it needs to be manually invalidated
    # to get up-to-date listings
    fs.invalidate_cache()
    try:
        segments = sorted(fs.ls(artifacts_dir))
    except FileNotFoundError:
        segments = []

    if len(segments) > 0:
        return vcm.to_url(fs, segments[-1])


def read_run_config(run_url):
    fs = vcm.get_fs(run_url)
    s = fs.cat(os.path.join(run_url, "fv3config.yml"))
    return fv3config.load(io.BytesIO(s))


def append_segment_to_run_url(run_url, wrapper):
    """Append an segment to an initialized segmented run

    Either runs the first segment, or additional ones
    """
    with tempfile.TemporaryDirectory() as dir_:
        print(f"Iteration run={run_url} working_directory={dir_}", file=sys.stderr)

        config = read_run_config(run_url)
        model_urls = get_model_urls(config)
        StepMetadata(
            job_type="prognostic_run",
            url=run_url,
            dependencies={"ml_models": model_urls} if len(model_urls) > 0 else None,
        ).print_json()
        last_segment = read_last_segment(run_url)

        if last_segment is not None:
            print("Continuing from segment", last_segment)
            config = fv3config.enable_restart(
                config, os.path.join(last_segment, "RESTART")
            )
        else:
            print(f"First segment in {run_url}")

        rundir = os.path.join(dir_, "rundir")
        post_processed_out = os.path.join(dir_, "post_processed")

        exit_code = run_segment(config, rundir, wrapper)
        if exit_code != 0:
            warnings.warn(
                UserWarning(f"FV3 exited with a nonzero exit-code: {exit_code}")
            )
        preexisting_files = os.path.join(rundir, "preexisting_files.txt")
        print("Skipping upload of the following files:")
        with open(preexisting_files) as f:
            print(f.read())

        post_process(
            rundir=rundir,
            destination=post_processed_out,
            skip=preexisting_files,
            chunks=os.path.join(rundir, "chunks.yaml"),
        )

        append_segment(
            rundir=post_processed_out,
            destination=run_url,
            segment_label=None,
            no_copy=True,
        )
        print("Cleaning up working directory")
        return exit_code
