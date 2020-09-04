.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################
VERSION ?= $(shell git rev-parse HEAD)
ENVIRONMENT_SCRIPTS = .environment-scripts
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = fv3net
PYTHON_INTERPRETER = python3
DATA = data/interim/advection/2019-07-17-FV3_DYAMOND_0.25deg_15minute_regrid_1degree.zarr.dvc
IMAGE = fv3net
GCR_IMAGE = us.gcr.io/vcm-ml/fv3net

GCR_BASE  = us.gcr.io/vcm-ml
FV3NET_IMAGE = $(GCR_BASE)/fv3net
PROGNOSTIC_RUN_IMAGE = $(GCR_BASE)/prognostic_run


ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: build_images push_image run_integration_tests image_name_explicit

# pattern rule for building docker images
build_image_%:
	docker build . -f docker/$*/Dockerfile  -t $*

build_image_post_process_run:
	docker build workflows/post_process_run -t post_process_run

enter_%:
	docker run -ti -w /fv3net -v $(shell pwd):/fv3net $* bash

build_images: build_image_fv3net build_image_prognostic_run build_image_post_process_run

push_images: push_image_prognostic_run push_image_fv3net push_image_post_process_run

push_image_%:
	docker tag $* $(GCR_BASE)/$*:$(VERSION)
	docker push $(GCR_BASE)/$*:$(VERSION)

pull_image_%:
	docker pull $(GCR_BASE)/$*:$(VERSION)

enter: build_image
	docker run -it -v $(shell pwd):/code \
		-e GOOGLE_CLOUD_PROJECT=vcm-ml \
		-w /code $(IMAGE)  bash

#		-e GOOGLE_APPLICATION_CREDENTIALS=/google_creds.json \
#		-v $(HOME)/.config/gcloud/application_default_credentials.json:/google_creds.json \

build_ci_image:
	docker build -t us.gcr.io/vcm-ml/circleci-miniconda-gfortran:latest - < .circleci/dockerfile


# run integration tests
run_integration_tests:
	./tests/end_to_end_integration/run_test.sh $(VERSION)

test:
	pytest external tests

test_prognostic_run:
	docker run prognostic_run pytest

test_unit:
	coverage run -m pytest -m "not regression" --mpl --mpl-baseline-path=tests/baseline_images

test_regression:
	coverage run -m pytest -vv -m regression -s

test_dataflow:
	coverage run -m pytest -vv workflows/dataflow/tests/integration -s

coverage_report:
	coverage report -i --omit='**/test_*.py',conftest.py,'external/fv3config/**.py','external/fv3util/**.py'

htmlcov:
	rm -rf $@
	coverage html -i --omit='**/test_*.py',conftest.py,'external/fv3config/**.py','external/fv3util/**.py'

test_argo:
	make -C workflows/argo/ test

## Make Dataset
.PHONY: data update_submodules create_environment overwrite_baseline_images
data:
	dvc repro $(DATA)

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Set up python interpreter environment
update_submodules:
	git submodule sync --recursive
	git submodule update --recursive --init


install_deps:
	bash $(ENVIRONMENT_SCRIPTS)/build_environment.sh $(PROJECT_NAME)

lock_deps:
	conda-lock -f environment.yml
	pip-compile pip-requirements.txt external/vcm/setup.py docker/**/requirements.txt --output-file constraints.txt

install_local_packages:
	bash $(ENVIRONMENT_SCRIPTS)/install_local_packages.sh $(PROJECT_NAME)

create_environment:
	bash $(ENVIRONMENT_SCRIPTS)/build_environment.sh $(PROJECT_NAME)
	bash $(ENVIRONMENT_SCRIPTS)/install_local_packages.sh $(PROJECT_NAME)


overwrite_baseline_images:
	pytest tests/test_diagnostics_plots.py --mpl-generate-path tests/baseline_images

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

snakemake_k8s: push_image
	make -C k8s-workflows/scale-snakemake/

snakemake:
	bash -c 'snakemake 2> >(tee snakemake_log.txt)'


PYTHON_FILES = $(shell git ls-files | grep -e 'py$$' | grep -v -e '__init__.py')
PYTHON_INIT_FILES = $(shell git ls-files | grep '__init__.py')

check_file_size:
	./tests/check_for_large_files.sh

typecheck:
	./check_types.sh

lint: check_file_size
	black --diff --check $(PYTHON_FILES) $(PYTHON_INIT_FILES)
	flake8 $(PYTHON_FILES)
	# ignore unused import error in __init__.py files
	flake8 --ignore=F401 E203 $(PYTHON_INIT_FILES)
	@echo "LINTING SUCCESSFUL"

reformat:
	black $(PYTHON_FILES) $(PYTHON_INIT_FILES)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
