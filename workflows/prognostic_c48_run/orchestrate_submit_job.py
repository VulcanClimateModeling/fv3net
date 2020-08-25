import argparse
import os
import yaml
import logging
from pathlib import Path
import copy

import fv3config
import fv3kube
import vcm

logger = logging.getLogger(__name__)
PWD = Path(os.path.abspath(__file__)).parent
RUNFILE = os.path.join(PWD, "sklearn_runfile.py")
CONFIG_FILENAME = "fv3config.yml"
MODEL_FILENAME = "sklearn_model.pkl"


def get_kube_opts(config_update, image_tag=None):
    default = {
        "gcp_secret_name": "gcp-key",
        "post_process_image": "us.gcr.io/vcm-ml/post_process_run",
        "fv3_image": "us.gcr.io/vcm-ml/prognostic_run",
        "fv3config_image": "us.gcr.io/vcm-ml/prognostic_run",
    }
    kube_opts = copy.copy(default)
    kube_opts.update(config_update.get("kubernetes", {}))
    if args.image_tag:
        for key in ["fv3config_image", "fv3_image", "post_process_image"]:
            kube_opts[key] += ":" + args.image_tag
    return kube_opts


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "initial_condition_url",
        type=str,
        help="Remote url to directory holding timesteps with model initial conditions.",
    )
    parser.add_argument(
        "ic_timestep",
        type=str,
        help="Time step to grab from the initial conditions url.",
    )
    parser.add_argument(
        "output_url",
        type=str,
        help="Remote storage location for prognostic run output.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        help="Remote url to a trained sklearn model.",
    )
    parser.add_argument(
        "--nudge-to-observations", action="store_true", help="Nudge to observations",
    )
    parser.add_argument(
        "--image-tag", type=str, default=None, help="tag to apply to all default images"
    )
    parser.add_argument(
        "--prog_config_yml",
        type=str,
        default="prognostic_config.yml",
        help="Path to a config update YAML file specifying the changes from the base"
        "fv3config (e.g. diag_table, runtime, ...) for the prognostic run.",
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    parser.add_argument(
        "-a",
        "--allow_fail",
        action="store_true",
        help="Do not raise error if job fails.",
    )
    parser.add_argument(
        "-d",
        "--detach",
        action="store_true",
        help="Do not wait for the k8s job to complete.",
    )

    return parser


def insert_sklearn_settings(model_config, model_url):
    # Add scikit learn ML model config section
    scikit_learn_config = model_config.get("scikit_learn", {})
    scikit_learn_config["zarr_output"] = "diags.zarr"
    if model_url:

        # insert the model asset
        model_url = os.path.join(args.model_url, MODEL_FILENAME)
        model_asset = fv3config.get_asset_dict(
            args.model_url, MODEL_FILENAME, target_name="model.pkl"
        )
        model_config.setdefault("patch_files", []).append(model_asset)

        scikit_learn_config.update(
            model="model.pkl", diagnostic_ml=args.diagnostic_ml,
        )

    model_config["scikit_learn"] = scikit_learn_config


def insert_default_diagnostics(model_config):
    """ Inserts default diagnostics save configuration into base config,
    which is overwritten by user-provided config if diagnostics entry is present.
    Defaults to of saving original 2d diagnostics on 15 min frequency.
    If variables in this list does not exist, e.g. *_diagnostic vars only exist
    if --diagnostic_ml flag is set, they are skipped.

    Args:
        model_config: Prognostic run configuration dict
    """
    model_config["diagnostics"] = [
        {
            "name": "diags.zarr",
            "variables": [
                "net_moistening",
                "net_moistening_diagnostic",
                "net_heating",
                "net_heating_diagnostic",
                "water_vapor_path",
                "physics_precip",
            ],
            "times": {"kind": "interval", "frequency": 900},
        }
    ]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()

    # Get model config with prognostic run updates
    with open(args.prog_config_yml, "r") as f:
        user_config = yaml.safe_load(f)

    # It should be possible to implement all configurations as overlays
    # so this could be done as one vcm.update_nested_dict call
    # updated_nested_dict just needs to know how to merge patch_files fields
    config = vcm.update_nested_dict(
        fv3kube.get_base_fv3config(user_config.get("base_version")),
        fv3kube.c48_initial_conditions_overlay(
            args.initial_condition_url, args.ic_timestep
        ),
        {"diag_table": "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"},
    )
    insert_sklearn_settings(config, args.model_url)
    insert_default_diagnostics(config)

    model_config = vcm.update_nested_dict(
        config,
        # User settings override previous ones
        user_config,
    )

    if args.nudge_to_observations:
        model_config = fv3kube.enable_nudge_to_observations(model_config)

    # submission scripts
    short_id = fv3kube.get_alphanumeric_unique_tag(8)
    job_label = {
        "orchestrator-jobs": f"prognostic-group-{short_id}",
        # needed to use pod-disruption budget
        "app": "end-to-end",
    }
    kube_opts = get_kube_opts(user_config, args.image_tag)
    pod_spec = fv3kube.containers.post_processed_fv3_pod_spec(
        model_config, args.output_url, **kube_opts
    )
    job = fv3kube.containers.pod_spec_to_job(
        pod_spec, labels=job_label, generate_name="prognostic-run-"
    )

    client = fv3kube.initialize_batch_client()
    created_job = client.create_namespaced_job("default", job)
    logger.info(f"Created job: {created_job.metadata.name}")

    if not args.detach:
        fv3kube.wait_for_complete(job_label, raise_on_fail=not args.allow_fail)
        fv3kube.delete_completed_jobs(job_label)
