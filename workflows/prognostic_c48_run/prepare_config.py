import argparse
import os
import yaml
import logging

import fv3config
import fv3kube

import vcm

logger = logging.getLogger(__name__)


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "user_config",
        type=str,
        help="Path to a config update YAML file specifying the changes from the base"
        "fv3config (e.g. diag_table, runtime, ...) for the prognostic run.",
    )
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
        "--model_url", type=str, default=None, help="Remote url to a trained ML model.",
    )
    parser.add_argument(
        "--nudge-to-observations", action="store_true", help="Nudge to observations",
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    return parser


def ml_settings(model_type, model_url):
    if not model_url:
        return {}
    elif model_type == "scikit_learn":
        return sklearn_overlay(model_url)
    elif model_type == "keras":
        return keras_overlay(model_url)
    else:
        return {}


def sklearn_overlay(model_url):
    return {"scikit_learn": {"model": model_url}}


def keras_overlay(model_url, keras_dirname="model_data"):
    model_asset_list = fv3config.asset_list_from_path(
        os.path.join(model_url, keras_dirname), target_location=keras_dirname
    )
    return {"patch_files": model_asset_list, "scikit_learn": {"model": keras_dirname}}


def diagnostics_overlay(diagnostic_ml):
    return {
        "diagnostics": [
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
        ],
        "diag_table": "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic",
        "scikit_learn": {"diagnostic_ml": diagnostic_ml},
    }


def prepare_config(
    user_config,
    ic_timestep,
    initial_condition_url,
    diagnostic_ml=None,
    model_url=None,
    nudge_to_observations=False,
) -> dict:

    model_type = user_config.get("scikit_learn", {}).get("model_type", "scikit_learn")

    # get timing information
    duration = fv3config.get_run_duration(user_config)
    current_date = vcm.parse_current_date_from_str(ic_timestep)

    # To simplify the configuration flow, updates should be implemented as
    # overlays (i.e. diffs) requiring only a small number of inputs. In
    # particular, overlays should not require access to the full configuration
    # dictionary.
    overlays = [
        fv3kube.get_base_fv3config(user_config.get("base_version")),
        fv3kube.c48_initial_conditions_overlay(initial_condition_url, ic_timestep),
        diagnostics_overlay(diagnostic_ml),
        ml_settings(model_type, model_url),
        user_config,
    ]

    if nudge_to_observations:
        overlays.append(
            fv3kube.enable_nudge_to_observations(
                duration,
                current_date,
                nudge_url="gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data",
            )
        )

    return fv3kube.merge_fv3config_overlays(*overlays)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()

    with open(args.user_config, "r") as f:
        user_config = yaml.safe_load(f)

    print(
        yaml.safe_dump(
            prepare_config(
                user_config,
                args.ic_timestep,
                args.initial_condition_url,
                args.diagnostic_ml,
                args.model_url,
                args.nudge_to_observations,
            )
        )
    )
