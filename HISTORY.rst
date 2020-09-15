=======
History
=======

latest
------
* Extended :code:`workflows/prognostic_c48_run` to run with :code:`fv3fit.keras` ML models
* Unified :code:`fv3fit` API between keras and sklearn subpackages; offline ml diagnostics enabled for keras models
* Delete workflows/end_to_end
* Delete workflows/one_step_jobs
* Delete workflows/one_step_diags
* Delete loaders/mappers/_one_step.py
* Delete fv3net/pipelines/create_training_data
* Delete workflows/create_training_data
* Delete tests/end_to_end_integration
* Rename tests/end_to_end_integration_argo to tests/end_to_end_integration
* Delete run_argo_integration_tests rule from fv3net Makefile
* Add new end-to-end integration test that uses nudged-to-shield run to generate training data and argo for orchestration
* Move code in fv3net.regression to new package `fv3fit`, except for SklearnPredictionMapper which is moved to `workflows/offline_ml_diags`
* Add -<dQ2> to physics precipitation within prognostic run runfile.
* Add regression tests for model training and offline ML diagnostic steps
* Stop e2e orchestrator from adding `extra_package` and `setup_file` arguments to Dataflow jobs. This allows use of `./dataflow.sh` script in e2e workflows.
* Increase nudging run memory limits and add a high-capacity dynamic volume for nudging output
* Add fixtures and common units tests for ML and diagnostic data mappers
* Add diurnal cycle functions to diagnostics_utils
* Add cos_zenith_angle calculation to vcm, if cos_zenith_angle provided as an input variable in training or diagnostic configuration, it will be derived and inserted into the batched data.
* Add general purpose transformations of mappers `ValMap` and `KeyMap`
* Add offset_seconds argument to the nudging loader
* Add diagnostic variables from SHiELD and FV3GFS to new diagnostic mappers for each data sources; diagnostic mappers return net_heating and net_precipitation for the coarsened SHiELD and coarse FV3GFS diagnostics
* Add surface type dimension to diurnal cycle calculated in diagnostics
* Remove renaming of time and data variables from batch functions in external/loaders. Renaming should be done by the mappers.
* Add nudge-to-obs runfile and argo workflow.
* Generalize post-processing script and create new post_process_run image.
* Move plotting routines previously found in :code:`vcm.visualize` and :code:`gallery` into a new package :code:`fv3viz` and remove dependencies between :code:`vcm` and plotting routines.
* Add vertical profiles of RMSE/bias to the offline diagnostics dataset.
* add new `v0.5` base fv3config to cope with the modification to how fv3config handles the location of the orographic data
* Fix partitioning of timesteps in loaders.batches so that it uses all timesteps, allows last batch to be < timesteps_per_batch.
* Add SHiELD and FV3GFS diagnostic variables such as :code:`net_heating` to the training data and offline ML diagnostics datasets, and compute composites over positive and negative :code:`net_precipitation` domains in these diagnostics
* Offline diagnostics report now uses the same config file as the training data.
* Use run-fv3gfs workflow template to do prognostic run
* Move orchestrate_submit_jobs.py to prepare_config.py and add regression test
* Prognostic and nudge-to-fine workflows now require fv3gfs-images at or after commit 46b1d8742ee220ebeb14942e0a0f43da963cc0cf of fv3gfs-wrapper.
* Make time labels of the wrapper diagnostics in the nudge-to-fine workflow consistent with the fortran diagnostics.

0.2.3 (2020-06-19)
------------------
* Gratuitous bug fixes
* Still upload failed prognostic runs (#416)
* Offline ML diagnostics from mappers

0.2.2 (2020-06-18)
------
* Add flag --allow-fail to prognostic run `orchestrate_submit_job.py` so prognostic run crashes don't have to kill e2e workflow
* makefile target for testing prognostic run: `make test_prognostic_run`
* Rename dimensions and variable names in input/output of prognostic run, new API for prognostic_run yamls
* Add option to not apply ML predictions to model state in prognostic runs (so-called "piggy-back" runs)
* Modify submission of prognostic run so that its configuration is specified independently of one-step jobs
* Add `base_fv3config_version` parameter to one-step and prognostic run yamls
* Add new `v0.4` base fv3config which is a configuration set up for being initialized from coarsened SHiELD restart fields
* Modify format of one-step yamls to not include explicit fv3config key (making them consistent with prognostic run yaml)
* Add a batched loader for nudging data using FunctionOutputSequence interface
* Refactor training data batch loading to use a common batch loader for all data sources. To load from a specific
 data source (e.g. one step, nudging), the function name `open_<data source type>` (selected from the available functions in loaders.__init__)
 should be specified in the training configuration YAML. Works for the currently existing mappers: one step and fine res.
* Add a `diagnostics_utils` poetry package in `./external` and a `training_data_diagnostics` workflow step for processing
training data from multiple sources into a set of diagnostic variables
* Add a `diagnostic_sequence_from_mapper` to the `loaders` module function for loading data for diagnostics
* Add nudging data loaders for use in diagnostics and ML training
* Add a mapper that takes in a model and any base mapper and includes ML prediction
* Add optional arg `--timesteps-file` to fv3net.regression.sklearn to provide a list of timesteps to use
 in conjunction with the config arg `num_timesteps_per_batch`. The training config arg `num_batches` is now
 deprecated in favor of providing a list of timesteps.
* Remove usage of "tmp_dataflow" directory from `vcm.cloud.gcs` testing infrastructure and skip extract tests
* Add end-to-end orchestration plugin point for nudged simulations
* Update prognostic run report to include global averages of physics variables and heating/moistening
* Prognostic report requires that prognostic runs have been post-processed before being passed to the report workflow
* Add a new transform mapper class `NudgedFullTendencies` which computes the missing pQ terms to give a full dataset of Q terms for the nudged source; uses existing mapping transforms `MergedNudged` and `NudgedStateCheckpoints`; new helper function to open the `NudgedFullTendencies` public nudged class
* Add arguments to rename dataset variables and dimensions in the nudged and fine-res helper functions and mapper classes to avoid renaming via the batch functions
* Adds ability to specify timestep offsets in the `FineResolutionSources` mapper
* Adds a regression test for the `training_data_diags` workflow step
* Add workflow for producing diagnostics of ML predicted dQ1 & dQ2 (workflows/offline_ml_diags)
* Add mapper that merges any two other mappers and concats overlapping variables along overlap dimension.


0.2.1 (2020-05-15)
------
* Add surface_precipitation_rate to one-step outputs, create training and test steps.
* Correct prognostic runfile diagnostic calculations.
* Update fv3gfs-python to v0.4.3.
* Updated fv3gfs-python to v0.4.1. As part of this, refactored sklearn_interface functions from runtime to the prognostic run runfile.
* Prognostic run report: compute and plot scalar metrics, generate report via
  python script, change output location (#226)
* Multithreaded uploading in one-step jobs (#260)
* Made nudging run upload more robust using k8s yaml templating submission and gsutil container upload
* The key for commands in the end to end config YAML can be given as either `command` or `argo`, and the arguments will be parsed into the appropriate format for either type.
* Offline diags workflow now downloads the test data to a local temp dir to speed reading and prevent remote read errors.

0.2.0 (2020-04-23)
------------------
* Added physics on end_to_end workflow configuration yaml. Only does baseline run for now.
* Added integration tests (tests/end_to_end_integration) that through CircleCI after image builds
* Fixed integration tests to use same version tags of the `fv3net` and `prognostic_run` images
* Added makefile targets to submit integration tests to cluster from local machine and to get docker image names
* Made simple step output directory names the default in the orchestrator
* Add `run_with_learned_nudging` workflow
* Update fv3config submodule to v0.3.1
* Add `get_config()` function to fv3net.runtime
* Change API of `diagnostics_to_zarr` workflow so that it saves output zarrs in the given run directory
* Add `nudge_to_obs` module to `kube_jobs`, which helps with the configuration of FV3GFS model runs that are nudged towards GFS analysis
* Add public function: vcm.convert_timestamps
* Add pipeline to load C384 restart data into a zarr
* One step run workflow outputs a single zarr as output (instead of individual directories for each timestep), downstream workflows are adjusted to use this data format
* Train data pipeline and offline diagnostics workflow read in variable names information from yaml provided to python
* Force load data in diagnostics workflow before compute and plot
* Improved logging when running FV3 model
* HTML reports now have title and timestamp, and optionally can include a dict of metadata as a table
* `test_sklearn_model` and `train_sklearn_model` workflows save a yaml of all the timesteps for each respective step
* `train_sklearn_model` now creates an html report of its own, which includes ML model metadata and a plot of temporal distribution of training data
* offline ML report generated by `test_sklearn_model` has new plot of temporal distribution of testing data
* new external package `report` created, which handles generation of workflow reports
* new external package `gallery` created, which generates figures which can be used by multiple workflows
* add __main__.py to fv3net/regression/sklearn in order to better separate model training from I/O and report generation
* Build `prognostic_run` image from v0.3.5 of `fv3gfs-python`
* Adjust diagnostic outputs for prognostic run with name net_moistening instead of net_precip and add total_precipitation to outputs
* final adjustments and fixes for advisory council results



0.1.1 (2020-03-25)
------------------
* Updates to make end-to-end workflow work with fv3atm (fv3gfs-python:v0.3.1)
* Added bump2version for automated versioning of `fv3net` resources
* Add CircleCI build/push capabilities for `fv3net` images


0.1.0 (2020-03-20)
------------------
* First release of fv3net
