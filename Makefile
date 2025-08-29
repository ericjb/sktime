# Makefile for easier installation and cleanup.
#
# Uses self-documenting macros from here:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

PACKAGE=sktime
DOC_DIR=./docs
BUILD_TOOLS=./build_tools
TEST_DIR=testdir

.PHONY: help release install test lint clean dist doc docs

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

release: ## Make a release
	python3 $(BUILD_TOOLS)/make_release.py

install: ## Install for the current user using the default python command
	python3 setup.py build_ext --inplace && python setup.py install --user

test: ## Run unit tests
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	python -m pytest

full_test: ## Run all tests
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	python -m pytest --only_changed_modules False

test_without_datasets: ## Run unit tests skipping sktime/datasets
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	python -m pytest --ignore sktime/datasets

test_check_suite: ## run only estimator contract tests in TestAll classes
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	python -m pytest -k 'TestAll' $(PYTESTOPTIONS)

test_softdeps: ## Run unit tests to check soft dependency handling in estimators
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	cd ${TEST_DIR}
	python -m pytest -v -n auto --showlocals -k 'test_all_estimators' $(PYTESTOPTIONS) --pyargs sktime.registry
	python -m pytest -v -n auto --showlocals -k 'test_check_estimator_does_not_raise' $(PYTESTOPTIONS) --pyargs sktime.utils.tests
	python -m pytest -v -n auto --showlocals $(PYTESTOPTIONS) --pyargs sktime.tests.test_softdeps

test_softdeps_full: ## Run all non-suite unit tests without soft dependencies or downloading datasets
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	cd ${TEST_DIR}
	python -m pytest -v --showlocals --ignore sktime/datasets -k 'not TestAll' $(PYTESTOPTIONS)

test_mlflow: ## Run mlflow integration tests
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	cd ${TEST_DIR}
	python -m pytest -v --showlocals $(PYTESTOPTIONS) --pyargs sktime.utils.tests.test_mlflow_sktime_model_export

tests: test

clean: ## Clean build dist and egg directories left after install
	rm -rf ./dist
	rm -rf ./build
	rm -rf ./pytest_cache
	rm -rf ./htmlcov
	rm -rf ./junit
	rm -rf ./$(PACKAGE).egg-info
	rm -rf coverage.xml
	rm -f MANIFEST
	rm -rf ./wheelhouse/*
	find . -type f -iname "*.so" -delete
	find . -type f -iname '*.pyc' -delete
	find . -type d -name '__pycache__' -empty -delete

dist: ## Make Python source distribution
	python3 setup.py sdist bdist_wheel

build:
	python -m build --sdist --wheel --outdir wheelhouse

docs: doc

doc: ## Build documentation with Sphinx
	$(MAKE) -C $(DOC_DIR) html

nb: clean
	rm -rf .venv || true
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install .[all_extras,binder] && ./build_tools/run_examples.sh

dockertest:
	docker build -t sktime -f build_tools/docker/$(PYTHON_VERSION).dockerfile .
	docker run -it --name sktime sktime bash -c "make full_test"

docker-pyfablearima: ## Build and run Docker image that executes check_estimator on PyFableARIMA
	docker build -t sktime-pyfablearima -f Dockerfile.pyfablearima .
	docker run --rm sktime-pyfablearima

docker-pyfablearima-pytest: ## Build image and run only the PyFableARIMA unit test via pytest inside container
	docker build -t sktime-pyfablearima -f Dockerfile.pyfablearima .
	docker run --rm -e CHECK_CMD="pytest -q sktime/forecasting/tests/test_pyfable_arima.py::test_pyfablearima_formula_immutability" sktime-pyfablearima

docker-pyfablearima-shell: ## Build image and drop into interactive shell (R + Python ready)
	docker build -t sktime-pyfablearima -f Dockerfile.pyfablearima .
	docker run -it --rm -e CHECK_CMD="bash" sktime-pyfablearima

# ---------------------------------------------------------------------------
# Helper: generate Dockerfile.pyfablearima if missing (idempotent)
# Usage: make docker-pyfablearima-init
# Will NOT overwrite existing file.
docker-pyfablearima-init: ## Create Dockerfile.pyfablearima scaffold if absent
	@if [ -f Dockerfile.pyfablearima ]; then \
	  echo "Dockerfile.pyfablearima already exists; not overwriting"; \
	else \
	  echo "Creating Dockerfile.pyfablearima"; \
	  cat > Dockerfile.pyfablearima <<'EOF'; \
############################################################
# Auto-generated minimal Dockerfile for PyFableARIMA checks
############################################################
FROM rocker/r2u:jammy

ENV DEBIAN_FRONTEND=noninteractive \
		TZ=UTC \
		LANG=C.UTF-8 \
		LC_ALL=C.UTF-8 \
		PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
		python3 python3-venv python3-dev python3-pip build-essential git && \
		rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# Core Python deps (pin rpy2 for R 4.4 compatibility if needed)
RUN pip install --no-cache-dir numpy pandas scipy patsy rpy2

# R packages (fable stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
		r-cran-fable r-cran-fabletools r-cran-fpp3 r-cran-tsibble \
		r-cran-tidyverse r-cran-distributional r-cran-lubridate r-cran-dplyr && \
		rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml setup.cfg README.md ./
RUN pip install --no-cache-dir . pytest || true
COPY . /workspace
RUN pip install --no-cache-dir . pytest

CMD ["bash","-lc","python - <<'PY'\nfrom sktime.utils import check_estimator\nfrom sktime.forecasting.pyfable_arima import PyFableARIMA\nprint('Running check_estimator(PyFableARIMA)...')\ncheck_estimator(PyFableARIMA, raise_exceptions=True)\nprint('check_estimator passed.')\nPY"]
EOF; \
	fi
