import prepare_config
import pytest

MODEL_URL = "gs://ml-model"
IC_URL = "gs://ic-bucket"
IC_TIMESTAMP = "20160805.000000"
ML_CONFIG_UPDATE = "prognostic_config.yaml"
NUDGING_CONFIG_UPDATE = "nudge_to_fine_config.yaml"
OTHER_FLAGS = ["--nudge-to-observations"]


def get_ml_args():
    return [
        ML_CONFIG_UPDATE,
        IC_URL,
        IC_TIMESTAMP,
        "--model_url",
        MODEL_URL,
    ] + OTHER_FLAGS


def get_nudge_to_fine_args():
    return [
        NUDGING_CONFIG_UPDATE,
        IC_URL,
        IC_TIMESTAMP,
    ]


def test_prepare_ml_config_regression(regtest):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(get_ml_args())
    with regtest:
        prepare_config.prepare_config(args)


def test_prepare_nudge_to_fine_config_regression(regtest):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(get_nudge_to_fine_args())
    with regtest:
        prepare_config.prepare_config(args)


TIMESTAMPS = ["20160801.021500", "20160801.041500"]


@pytest.mark.parametrize(
    ["timestamps", "frequency_minutes", "expected"],
    [
        pytest.param(
            TIMESTAMPS, None, {"kind": "selected", "times": TIMESTAMPS}, id="timestamps"
        ),
        pytest.param(None, 120, {"kind": "interval", "frequency": 7200}, id="2-hourly"),
        pytest.param(
            None, None, {"kind": "interval", "frequency": 900}, id="default_15-minute"
        ),
    ],
)
def test_diagnostics_overlay_times(timestamps, frequency_minutes, expected):
    diags_overlay_times = prepare_config.diagnostics_overlay(
        {}, None, None, timestamps, frequency_minutes
    )["diagnostics"][0]["times"]
    assert diags_overlay_times == expected
