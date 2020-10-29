import prepare_config


def test_prepare_config_regression(regtest):

    example_config = {
        "base_version": "v0.4",
        "namelist": {
            "coupler_nml": {
                "days": 10,
                "hours": 0,
                "minutes": 0,
                "seconds": 0,
                "dt_atmos": 900,
                "dt_ocean": 900,
                "restart_secs": 0,
            },
            "atmos_model_nml": {"fhout": 0.25},
            "gfs_physics_nml": {"fhzero": 0.25},
            "fv_core_nml": {"n_split": 6},
        },
    }

    output = prepare_config.prepare_config(
        example_config,
        ic_timestep="20160805.000000",
        initial_condition_url="gs://ic-bucket",
        diagnostic_ml=False,
        model_url="gs://ml-model",
        nudge_to_observations=True,
    )

    print(output, file=regtest)


def test_prepare_config_regression_triggered(regtest):

    example_config = {
        "base_version": "v0.5",
        "diagnostics": [
            {
                "name": "data.zarr",
                "times": {"frequency": 7200, "kind": "interval"},
                "variables": [
                    "tendency_of_air_temperature_due_to_fv3_physics",
                    "tendency_of_specific_humidity_due_to_fv3_physics",
                    "air_temperature",
                    "specific_humidity",
                    "pressure_thickness_of_atmospheric_layer",
                ],
            }
        ],
        "forcing": "gs://vcm-fv3config/data/base_forcing/v1.1/",
        "orographic_forcing": "gs://vcm-fv3config/data/orographic_data/v1.0",
        "diag_table": "gs://vcm-ml-experiments/noah/prognostic_runs/2020-10-16-triggered-regressor/diag_table",  # noqa
        "namelist": {
            "coupler_nml": {
                "calendar": "julian",
                "days": 5,
                "hours": 0,
                "minutes": 0,
                "months": 0,
                "seconds": 0,
            },
            "gfdl_cloud_microphysics_nml": {"fast_sat_adj": False},
            "fv_core_nml": {"do_sat_adj": False},
            "gfs_physics_nml": {"ldiag3d": True, "fhzero": 2.0},
            "atmos_model_nml": {"fhout": 2.0, "fhmax": 10000},
        },
        "scikit_learn": {
            "model_type": "triggered_sklearn",
            "model_url": "gs://vcm-ml-archive/noah/emulator/2020-10-16-triggered-regressor",  # noqa
        },
    }

    output = prepare_config.prepare_config(
        example_config,
        ic_timestep="20160805.000000",
        initial_condition_url="gs://ic-bucket",
        diagnostic_ml=False,
        nudge_to_observations=True,
    )

    print(output, file=regtest)
