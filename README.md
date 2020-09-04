fv3net
==============================
[![CircleCI](https://circleci.com/gh/VulcanClimateModeling/fv3net.svg?style=svg&circle-token=98ccddae8375060a2fbbf240407dd4135d3dcf68)](https://circleci.com/gh/VulcanClimateModeling/fv3net)

Improving the GFDL FV3 model physics with machine learning

# The default fv3net environment

## Installing

This project specifies a default "fv3net" environment containing the
dependencies of all the non-containerized workflows. This environment uses
the conda package manager. If that is installed then you can run

    make update_submodules
    make create_environment

This creates an anaconda environment `fv3net` containing both Vulcan and
external dependencies. It downloads all the Vulcan submodules and installs them
in development mode (e.g. `pip install -e`) so that any modifications within
`external` will be loaded in the conda environment. Once this command completes
succesfully, the fv3net environment can be activated with

    conda activate fv3net

# Updating dependencies

To maintain deterministic builds, fv3net locks the versions of all pip and
anaconda packages.

# Pip

Pip dependencies are specified in a variety of places. Mostly `setup.py`
files and `requirements.txt` files for the dockerfiles. A package called
`pip-tools` is used to ensure that these files do not conflict with one
another. If they do not, the `make lock_deps` rule will generate a file
`constraints.txt` containing a list of versions to use for pip packages.
These constraints should then be whenenver `pip` is invoked like this:

    pip install -c constraints.txt <other pip args>

## Where to pin dependencies?

Since `constraints.txt` is compiled automatically, it should not be manually
edited. If you need to constrain or pin a dependency, you should do so in the
`requirements.txt` used by the build process for the container where the
problem occurs or in the root level `pip-requirements.txt` file. 

For instance, suppose `fsspec` v0.7.0 breaks some aspect of the prognostic
run image, then you could add something like the following to the
`docker/prognostic_run/requirements.txt`:

    fsspec!=0.7.0

Then run `make lock_deps` to update the `constraints.txt` file.

## The "fv3net" environment

The package `conda-lock` is used to ensure deterministic builds anaconda
builds. Therefore, adding or modifying a dependency involves a few steps:
1. add any anaconda packages to the `environment.yml`
1. add any pip packages to `pip-requirements.txt`
1. run `make lock_deps` to create lock files `conda-<system>.lock` which explicitly list all the conda packages
1. Commit the lock files and any other changes to git

The `make create_environment` uses these lock files and
`pip-requirements.txt` to install its dependencies. It does NOT directly
install the `environment.yml` file since that can lead to non-deterministic
builds, and difficult to debug errors in CI.

# Deploying cloud data pipelines

The main data processing pipelines for this project currently utilize Google Cloud
Dataflow and Kubernetes with Docker images.  Run scripts to deploy these workflows
along with information can be found under the `workflows` directory.

## Building the fv3net docker images

The workflows use a pair of common images:

|Image| Description| 
|-----|------------|
| `us.gcr.io/vcm-ml/prognostic_run` | fv3gfs-python with minimal fv3net and vcm installed |
| `us.gcr.io/vcm-ml/fv3net` | fv3net image with all dependencies including plotting |

These images can be built and pushed to GCR using `make build_images` and
`make push_images`, respectively.

## Running fv3gfs with Kubernetes

Docker images with the python-wrapped model and fv3run are available from the
[fv3gfs-python](https://github.com/VulcanClimateModeling/fv3gfs-python) repo.
Kubernetes jobs can be written to run the model on these docker images. A super simple
job would be to perform an `fv3run` command (provided by the
[fv3config package](https://github.com/VulcanClimateModeling/fv3config))
using google cloud storage locations. For example, running the basic model using a
fv3config dictionary in a yaml file to output to a google cloud storage bucket
would look like:

```
fv3run gs://my_bucket/my_config.yml gs://my_bucket/my_outdir
```

If you have a python model runfile you want to execute in place of the default model
script, you could use it by adding e.g. `--runfile gs://my-bucket/my_runfile.py`
to the `fv3run` command.

You could create a kubernetes yaml file which runs such a command on a
`fv3gfs-python` docker image, and submit it manually. However, `fv3config` also
supplies a `run_kubernetes` function to do this for you. See the
[`fv3config`](https://github.com/VulcanClimateModeling/fv3config) documentation for
more complete details.

The basic structure of the command is

    fv3config.run_kubernetes(
        config_location,
        outdir,
        docker_image,
        gcp_secret='gcp_key',
    )

Where `config_location` is a google cloud storage location of a yaml file containing
a fv3config dictionary, outdir is a google cloud storage location to put the resulting
run directory, `docker_image` is the name of a docker image containing `fv3config`
and `fv3gfs-python`, and `gcp_secret` is the name of the secret containing the google
cloud platform access key (as a json file called `key.json`). For our VCM group this
should be set to 'gcp_key'. Additional arguments are
available for configuring the kubernetes job and documented in the `run_kubernetes`
docstring.

# Code linting checks

This python code in this project is autoformated using the
[black](https://black.readthedocs.io/en/stable/) code formatting tool, and the
[isort](https://github.com/timothycrosley/isort) tool for automatically sorting
the order of import statements. To pass CI, any contributed code must be
unchanged by black and also checked by the flake8 linter. However, please use
isort to sort the import statements (done automatically by `make reformat`
below).

Contributers can see if their *commited* code passes these standards by running

    make lint

If it does not pass, than it can be autoformatted using 

    make reformat

# How to contribute to fv3net

Please see the [contribution guide.](./CONTRIBUTING.md)

# How to get updates and releases

For details on what's included or upcoming for a release, please see the [HISTORY.rst](./HISTORY.rst) document.

For instructions on preparing a release, please read [RELEASE.rst](./RELEASE.rst).

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
