## Argo Workflow Templates

Argo is a project for orchestrating containers, that we use for long-running
scientific workflows. This folder contains "WorkflowTempates" that can be
installed onto a K8s cluster. Once installed on a cluster, they can be
referenced from other argo workflows, or run directly using the `argo`
command line tool.

### Quickstart

To install these templates run

    kubectl apply -k <this directory>

This can be done from an external location (e.g. vcm-workflow-control). To run an installed workflowtemplate,
use the `--from` flag. Workflow parameters can be passed via the command line with the `-p` option. For example
```
argo submit --from workflowtemplate/prognostic-run-diags \
    -p runs="$(< rundirs.json)" \
    --name <name>
```

This job can be monitored by running

    argo get <name>

### Pinning the image tags

These workflows currently refer to following images without using any tags:
1. us.gcr.io/vcm-ml/fv3net
1. us.gcr.io/vcm-ml/prognostic_run
1. us.gcr.io/vcm-ml/post_process_run
1. us.gcr.io/vcm-ml/ufs-utils

However, you can and should pin these images using kustomize (>=v3). For
example, consuming configurations (e.g. in vcm-workflow-control) could use
the following kustomization.yaml to pin these versions:

```
apiVersion: kustomize.config.k8s.io/v1beta1
resources:
- <path/to/fv3net/workflows/argo>
kind: Kustomization
images:
- name: us.gcr.io/vcm-ml/fv3net
  newTag: 6e121e84e3a874c001b3b8d1b437813c9859e078
- name: us.gcr.io/vcm-ml/post_process_run
  newTag: 6e121e84e3a874c001b3b8d1b437813c9859e078
- name: us.gcr.io/vcm-ml/prognostic_run
  newTag: 6e121e84e3a874c001b3b8d1b437813c9859e078
- name: us.gcr.io/vcm-ml/ufs-utils
  newTag: 2023.03.06.4
```

It is also possible to do this programmatically, using `kustomize edit set image`.
See the [end-to-end integration tests](/tests/end_to_end_integration/run_test.sh) for an example.

### Running fv3gfs with argo

The `prognostic-run` template is a workflow to do fv3gfs simulations on the
cloud. It can do baseline (no-ML) runs, nudged runs or prognostic runs.
It does post-processing on the fly and the workflow can run the model in
sequential segments.

| Parameter            | Description                                                                   |
|----------------------|-------------------------------------------------------------------------------|
| `config`             | String representation of base config YAML file; supplied to prepare-config |
| `tag`                | Tag which describes the run and is used in its storage location               |
| `flags`              | (optional) Extra command line flags for prepare-config                    |
| `bucket`             | (optional) Bucket to save run output data; default 'vcm-ml-experiments'       |
| `project`            | (optional) Project directory to save run output data; default 'default'       |
| `segment-count`      | (optional) Number of prognostic run segments; default "1"                     |
| `cpu`                | (optional) Number of cpus to request; default "6"                             |
| `memory`             | (optional) Amount of memory to request; default 6Gi                           |
| `online-diags-flags` | (optional) `flags` for `prognostic-run-diags` workflow                        |
| `online-diags`       | (optional) Run online diagostics if "true"; default "true"                     |

#### Command line interfaces used by workflow
This workflow first resolves the output location for the run according to:
```
output="gs://{bucket}/{project}/$(date +%F)/{tag}/fv3gfs_run"
```
Slashes (`/`) are not permitted in `bucket`, `project` and `tag` to preserve the depth
of this directory structure.

And then calls:
```
prepare-config \
        {{inputs.parameters.flags}} \
        {{inputs.parameters.config}} \
        > /tmp/fv3config.yaml
```
And then
```
runfv3 create {output} /tmp/fv3config.yaml
```
Followed by `segment-count` iterations of
```
runfv3 append {output}
```

#### Restarting prognostic runs

Once a prognostic run has finished running, it is possible to extend the run using the
`restart-prognostic-run` workflow. Note it is not possible to change any aspects of the
configuration when restarting runs.

| Parameter       | Description                                                  |
|-----------------|--------------------------------------------------------------|
| `url`           | Location of existing prognostic run.                         |
| `segment-count` | (optional) Number of additional segments to run; default "1" |
| `cpu`           | (optional) Number of cpus to request; default "6"            |
| `memory`        | (optional) Amount of memory to request; default 6Gi          |

### Prognostic run report

The `prognostic-run-diags` workflow template will generate reports for
prognostic runs. See this [example][1].

| Parameter               | Description                                                                      |
|-------------------------|----------------------------------------------------------------------------------|
| `runs`                  | A json-encoded list of {"name": ..., "url": ...} items                           |
| `recompute-diagnostics` | (optional) whether to recompute diags before making report. Defaults to "false". |
| `flags`                 | (optional) flags to pass to `prognostic_run_diags save` command.                 |

To specify what verification data use when computing the diagnostics, use the `--verification`
flag. E.g. specifying the argo parameter `flags=" --verification nudged_c48_fv3gfs_2016"` will use a
year-long nudged-to-obs C48 run as verification. By default, the `40day_may2020` simulation
is used as verification. Datasets in the [vcm catalog](/external/vcm/vcm/catalog.yaml) with
the `simulation` and `category` metadata tags can be used.

By default, the report workflow assumes that the necessary diagnostics have already been
computed for each of the specified runs. If this is not the case, or if you want to recompute
the diagnostics for any reason, use `recompute-diagnostics=true`.

#### Command line interfaces used by workflow
For each `run` in the `runs` JSON parameter, this workflow calls `prognostic_run_diags save`
and `prognostic_run_diags metrics` to compute the diagnostics and metrics. The outputs are saved
to `{run.url}_diagnostics`. The workflow then regrids the relevant diagnostics to a lat-lon grid
using `cubed-to-latlon.regrid-single-input`. And then optionally calls
```
prognostic_run_diags movie {{run.url}} {{run.url}}_diagnostics
```
Once these steps are completed, a report is generated with
```
prognostic_run_diags report-from-json rundiags.json gs://vcm-ml-public/argo/{{workflow.name}}
```
where `rundiags.json` is generated from the supplied `rundirs.json` assuming diagnostics
are available at `{run.url}_diagnostics`

#### Workflow Usage Example

Typically, `runs` will be stored in a json file (e.g. `rundirs.json`).
```
[
  {
    "url": "gs://vcm-ml-experiments/default/2021-05-25/tuned-mp-no-ml-rad/fv3gfs_run",
    "name": "tuned-microphysics"
  },
  {
    "url": "gs://vcm-ml-experiments/default/2021-05-25/control-mp-no-ml-rad/fv3gfs_run",
    "name": "control"
  }
]
```

You can create a report from this json file using the following command from a bash shell:
```
argo submit --from workflowtemplate/prognostic-run-diags \
    -p runs="$(< rundirs.json)" \
    --name <name>
```

If successful, the completed report will be available at
`gs://vcm-ml-public/argo/<name>/index.html`, where `<name>` is the name of the created
argo `workflow` resource. This can be accessed from a web browser using this link:

    http://storage.googleapis.com/vcm-ml-public/argo/<name>/index.html

[1]: http://storage.googleapis.com/vcm-ml-public/experiments-2020-03/prognostic_run_diags/combined.html


### training workflow

This workflow trains machine learning models.

| Parameter              | Description                                          |
|------------------------|------------------------------------------------------|
| `input`                | Location of dataset for training data                |
| `config`               | Model training config yaml                           |
| `times`                | JSON-encoded list of timestamps to use for test data |
| `offline-diags-output` | Where to save offline diagnostics                    |
| `report-output`        | Where to save report                                 |
| `cpu`                  | (optional) # cpu for workflow. Defaults to 1.        |
| `memory`               | (optional) memory for workflow. Defaults to 6Gi.     |

#### Command line interfaces used by workflow
This workflow calls
```
python -m fv3fit.train \
          {{inputs.parameters.input}} \
          {{inputs.parameters.config}} \
          {{inputs.parameters.output}} \
          --timesteps-file {{inputs.parameters.times}} \
          {{inputs.parameters.flags}}
```

### offline-diags workflow

This workflow computes offline ML diagnostics and generates an associated report.

| Parameter              | Description                                          |
|------------------------|------------------------------------------------------|
| `ml-model`             | URL to machine learning model                        |
| `times`                | JSON-encoded list of timestamps to use for test data |
| `offline-diags-output` | Where to save offline diagnsostics                   |
| `report-output`        | Where to save report                                 |
| `memory`               | (optional) memory for workflow. Defaults to 6Gi.     |

#### Command line interfaces used by workflow
This workflow calls
```
python -m fv3net.diagnostics.offline.compute \
          {{inputs.parameters.ml-model}} \
          {{inputs.parameters.offline-diags-output}} \
          --timesteps-file {{inputs.parameters.times}}

python -m fv3net.diagnostics.offline.views.create_report \
          {{inputs.parameters.offline-diags-output}} \
          {{inputs.parameters.report-output}} \
          --commit-sha "$COMMIT_SHA"
```

### train-diags-prog workflow template

This workflow template runs the `training`, `offline-diags`, `prognostic-run` and
`prognostic-run-diags.diagnostics-step` workflow templates in sequence.

This workflow takes a `training-configs` parameter which is the string representation of a JSON file that should be
formatted as `[{name: model_name, config: model_config}, ...]`, and where the individal model config values are
as for the `training` workflow.  In practice it may be easiest to write this as a YAML file and then converted to
JSON format using `yq . config.yml` in a submission script.

The default behavior for the final `prognostic-run-diags.diagnostics-step` is to use the default verification dataset
(`40day_may2020`) to calculate the metrics. If this is not the appropriate verification data to use, make sure to specify
the appropriate verification using the `online-diags-flags` parameter, e.g. `-p online-diags-flags="--verification <name>"`.


| Parameter               | Description                                                           |
|-------------------------|-----------------------------------------------------------------------|
| `tag`                   | Tag which describes the experiment and is used in its storage location|
| `train-test-data`       | `input` for `training` workflow                                       |
| `training-configs`      | List of dicts of type `{name: config}`, where `config` is the config used for `training` workflow |
| `train-times`           | `times` for `training` workflow                                       |
| `test-times`            | `times` for `offline-diags` workflow                                  |
| `public-report-output`  | `report-output` for `offline-diags` workflow                          |
| `prognostic-run-config` | `config` for `prognostic-run` workflow                                |
| `bucket`                | (optional) Bucket to save output data; default 'vcm-ml-experiments'   |
| `project`               | (optional) Project directory to save output data; default 'default'   |
| `flags`                 | (optional) `flags` for `prognostic-run` workflow                      |
| `segment-count`         | (optional) `segment-count` for `prognostic-run` workflow; default "1" |
| `cpu-prog`              | (optional) `cpu` for `prognostic-run` workflow; default "6"           |
| `memory-prog`           | (optional) `memory` for `prognostic-run` workflow; default 6Gi        |
| `cpu-training`          | (optional) `cpu` for `training` workflow; default "1"                 |
| `memory-training`       | (optional) `memory` for `training` workflow; default 6Gi              |
| `memory-offline-diags`  | (optional) `memory` for `offline-diags` workflow; default 6Gi         |
| `training-flags`        | (optional) `flags` for `training` workflow                            |
| `online-diags-flags`    | (optional) `flags` for `prognostic-run-diags` workflow                |
| `do-prognostic-run`     | (optional) do prognostic run step; default "true"                     |
| `wandb-project`     | (optional) if --wandb flag provided to training, will log under this project ; default 'argo-default' |
| `wandb-tags`     | (optional) if --wandb flag provided to training, will log under these tags; default none |
| `wandb-group`     | (optional) if --wandb flag provided to training, will log under this group; default none    |


Output for the various steps will be written to `gs://{bucket}/{project}/$(date +%F)/{tag}`.
Slashes (`/`) are not permitted in `bucket`, `project` and `tag` to preserve the depth
of this directory structure.

### Cubed-sphere to lat-lon interpolation workflow

The `cubed-to-latlon` workflow can be used to regrid cubed sphere FV3 data using GFDL's `fregrid` utility.
In this workflow, you specify the input data (the prefix before `.tile?.nc`), the destination
for the regridded outputs, and a comma separated list of variables to regrid from the source file.

| Parameter       | Description                                                              | Example                         |
|-----------------|--------------------------------------------------------------------------|---------------------------------|
| `source_prefix` | Prefix of the source data in GCS (everything but .tile1.nc)              | gs://path/to/sfc_data (no tile) |
| `output-bucket` | URL to output file in GCS                                                | gs://vcm-ml-scratch/output.nc      |
| `resolution`    | Resolution of input data (defaults to C48)                               | one of 'C48', 'C96', or 'C384'  |
| `fields`        | Comma-separated list of variables to regrid                              | PRATEsfc,LHTFLsfc,SHTFLsfc      |
| `extra_args`    | Extra arguments to pass to fregrid. Typically used for target resolution | --nlat 180 --nlon 360           |

### Restart files to NGGPS initial condition workflow

The `chgres-cube` workflow can be used to transform a set of restart files to an
NGGPS-style initial condition with a new horizontal resolution.  It does so
using the [`UFS_UTILS`](https://github.com/ufs-community/UFS_UTILS)
`chgres_cube` tool. The workflow takes the following parameters:

| Parameter           | Description                                                                         | Example                                                                                                    |
|---------------------|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `restarts`          | Dirctory of restart files on GCS                                                    | gs://path/to/restarts                                                                                      |
| `date`              | Date of the restart files (YYYYMMDDHH)                                              | 2017010100                                                                                                 |
| `source_resolution` | Resolution of the restart files (defaults to C48)                                   | one of 'C48', 'C96', or 'C384'                                                                             |
| `target_resolution` | Resolution of the target initial condition (defaults to C384)                       | one of 'C48', 'C96', or 'C384'                                                                             |
| `tracers`           | Tracers included in the restart files                                               | '"sphum","liq_wat","o3mr","ice_wat","rainwat","snowwat","graupel","sgs_tke"'                               |
| `vcoord_file`       | Text file containing information about the vertical coordinate of the restart files | gs://vcm-ml-intermediate/2023-02-24-chgres-cube-hybrid-levels/global_hyblev.l63.txt                        |
| `reference_data`    | Path to forcing data on GCS (typically just use the default)                        | gs://vcm-ml-raw-flexible-retention/2023-02-24-chgres-cube-forcing-data/2023-02-24-chgres-cube-forcing-data |
| `destination_root`  | Path to store resulting initial condition                                           | gs://path/to/destination                                                                                   |
| `mpi_tasks`         | Number of MPI tasks to run with (default "6")                                       | "6"                                                                                                        |
| `memory`            | Memory to allocate (default "25Gi")                                                 | "25Gi"                                                                                                     |
