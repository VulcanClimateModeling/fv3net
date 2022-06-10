import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Tuple
from fv3net.artifacts.metadata import StepMetadata, log_fact_json
import numpy as np
import os
import tensorflow as tf

from fv3fit import set_random_seed
from fv3fit.train_microphysics import TrainConfig
from fv3fit._shared import put_dir
from fv3fit.emulation.keras import score_model
from fv3fit.wandb import (
    log_profile_plots,
    log_to_table,
)
from vcm import get_fs
import yaml
from emulation.config import ModelConfig

logger = logging.getLogger(__name__)


def load_final_model_or_checkpoint(train_out_url) -> Tuple[tf.keras.Model, str]:

    model_url = os.path.join(train_out_url, "model.tf")
    checkpoints = os.path.join(train_out_url, "checkpoints", "*.tf")

    fs = get_fs(train_out_url)
    if fs.exists(model_url):
        logger.info(f"Loading model for scoring from: {model_url}")
        url_to_load = model_url
    elif fs.glob(checkpoints):
        url_to_load = sorted(fs.glob(checkpoints))[-1]
        logger.info(f"Loading last model checkpoint for scoring from: {url_to_load}")
    else:
        raise FileNotFoundError(f"No keras models found at {train_out_url}")

    return tf.keras.models.load_model(url_to_load), url_to_load


def main(
    config: TrainConfig,
    seed: int = 0,
    model_url: str = None,
    emulation_config_path: Path = None,
):

    logging.basicConfig(level=getattr(logging, config.log_level))
    set_random_seed(seed)

    if config.use_wandb:
        d = asdict(config)
        d["model_url_override"] = model_url
        config.wandb.init(config=d)

    if model_url is None:
        model, model_url = load_final_model_or_checkpoint(config.out_url)
    else:
        logger.info(f"Loading user specified model from {model_url}")
        model = tf.keras.models.load_model(model_url)

    if emulation_config_path is not None:
        with emulation_config_path.open() as f:
            emu_config = ModelConfig.from_dict(yaml.safe_load(f))
    else:
        emu_config = None  # noqa

    StepMetadata(
        job_type="train_score",
        url=config.out_url,
        dependencies=dict(
            train_data=config.train_url, test_data=config.test_url, model=model_url
        ),
        args=sys.argv[1:],
    ).print_json()

    train_ds = config.open_dataset(
        config.train_url, config.nfiles, config.model_variables
    )
    test_ds = config.open_dataset(
        config.train_url, config.nfiles, config.model_variables
    )

    train_set = next(iter(train_ds.unbatch().shuffle(160_000).batch(80_000)))
    test_set = next(iter(test_ds.unbatch().shuffle(160_000).batch(80_000)))

    train_scores, train_profiles = score_model(model, train_set)
    test_scores, test_profiles = score_model(model, test_set)
    logger.debug("Scoring Complete")

    summary_metrics: Dict[str, Any] = {
        f"score/train/{key}": value for key, value in train_scores.items()
    }
    summary_metrics.update(
        {f"score/test/{key}": value for key, value in test_scores.items()}
    )

    # Logging for google cloud
    log_fact_json(data=summary_metrics)

    if config.use_wandb:
        pred_sample = model.predict(test_set, batch_size=8192)
        log_profile_plots(test_set, pred_sample)

        # add level for dataframe index, assumes equivalent feature dims
        sample_profile = next(iter(train_profiles.values()))
        train_profiles["level"] = np.arange(len(sample_profile))
        test_profiles["level"] = np.arange(len(sample_profile))

        config.wandb.job.log(summary_metrics)

        log_to_table("profiles/train", train_profiles)
        log_to_table("profiles/test", test_profiles)

    with put_dir(config.out_url) as tmpdir:
        with open(os.path.join(tmpdir, "scores.json"), "w") as f:
            json.dump({"train": train_scores, "test": test_scores}, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_url",
        help=(
            "Specify model path to run scoring for. Overrides use of models "
            "at the config.out_url"
        ),
        default=None,
    )
    parser.add_argument(
        "--emulation_model_config",
        type=Path,
        optional=True,
        help=("Load ModelConfig for post-hoc emulation corrections"),
    )

    known, unknown = parser.parse_known_args()
    config = TrainConfig.from_args(unknown)

    main(
        config,
        model_url=known.model_url,
        emulation_config_path=known.emulation_model_config,
    )
