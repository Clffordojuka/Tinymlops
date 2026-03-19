# tinyml-ops

A C-first machine learning systems project that combines a lightweight neural network core with reproducible training, experiment tracking, automated testing, and workflow-oriented MLOps practices.

## Overview

`tinyml-ops` started as a small neural network implementation in C and evolved into a compact end-to-end MLOps workflow.

The project currently supports:

- matrix operations
- dense layer forward and backward propagation
- ReLU activation
- MSE loss
- single-layer training with gradient updates
- CSV dataset loading
- config-driven training
- checkpoint save/load
- evaluation and prediction apps
- DVC-based reproducible pipelines
- experiment archiving and comparison
- MLflow tracking
- GitHub Actions CI
- Docker-based reproducible execution

## Repository structure

```text
tinyml-ops/
├── include/
│   └── tinyml.h
├── src/
│   ├── core/
│   ├── layers/
│   ├── optim/
│   ├── io/
│   ├── data/
│   └── utils/
├── apps/
│   ├── train/
│   ├── evaluate/
│   └── predict/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
├── configs/
│   ├── base/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── models/
│   ├── checkpoints/
│   ├── registry/
│   └── exported/
├── metrics/
├── logs/
├── docs/
├── .github/
│   └── workflows/
├── docker/
├── dvc.yaml
├── params.yaml
├── CMakeLists.txt
├── Makefile
├── README.md
└── .gitignore
````

## Recommended environment

The recommended execution environment is Docker.

Native Windows GCC builds were inconsistent during setup, while Docker/Linux provides a stable and reproducible workflow for:

* building
* testing
* running DVC pipelines
* running experiments
* comparing results
* logging runs to MLflow

## Build with Docker

### Build the image

```bash
docker build -f docker/Dockerfile.dev -t tinyml-ops-dev .
```

### Run the container

```bash
docker run --rm -it -v ${PWD}:/workspace tinyml-ops-dev
```

### Inside the container

```bash
cmake -S . -B build-docker -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build build-docker
ctest --test-dir build-docker --output-on-failure
```

## Core applications

The project currently provides three CLI applications:

* `train_app`
* `evaluate_app`
* `predict_app`

### Train

Trains the model from a config file and produces:

* checkpoint
* training metrics
* console training logs

### Evaluate

Loads a checkpoint and dataset, then reports:

* evaluation loss
* test prediction
* loaded parameters

### Predict

Loads a checkpoint and predicts a value for a single input.

## Configuration

Training and evaluation are controlled through config files.

Base config:

```text
configs/base/train_linear.cfg
```

Experiment configs:

```text
configs/experiments/linear_fast.cfg
configs/experiments/linear_long.cfg
```

These configs define values such as:

* dataset path
* epoch count
* learning rate
* metrics output path
* checkpoint output path

## Training

Train from a config file inside Docker:

```bash
./build-docker/apps/train_app configs/experiments/linear_long.cfg
```

Typical output includes:

* epoch-by-epoch loss
* final prediction for a sample input
* learned weight and bias

## Evaluation

Evaluate a trained checkpoint:

```bash
./build-docker/apps/evaluate_app configs/experiments/linear_long.cfg
```

Typical output includes:

* evaluation loss
* prediction for `x=4.0`
* loaded weight and bias

## Prediction

Run inference for a single input value:

```bash
./build-docker/apps/predict_app configs/base/train_linear.cfg 4.0
```

Example output includes:

* input value
* predicted output
* checkpoint path
* model weight and bias

## Dataset handling

The current project uses a small CSV regression dataset.

Example sample file:

```text
data/samples/linear.csv
```

Current dataset support is intentionally simple:

* one feature column
* one target column
* numeric CSV parsing
* no missing-value handling
* no batching abstraction yet

## DVC pipeline

The project includes a reproducible DVC pipeline for training and evaluation.

Run inside Docker:

```bash
python -m dvc repro
python -m dvc metrics show
```

This pipeline tracks:

* config-driven stage execution
* checkpoint output
* training metrics
* evaluation metrics

## Experiment runner

Use the experiment runner to automate config switching, DVC reproduction, and archiving:

```bash
python scripts/run_experiment.py configs/experiments/linear_fast.cfg
python scripts/run_experiment.py configs/experiments/linear_long.cfg
```

This script:

* updates `params.yaml`
* runs `dvc repro`
* reads train and eval metrics
* archives train metrics into `metrics/archive/`
* archives eval metrics into `metrics/archive/`
* archives checkpoints into `models/registry/`
* prints a concise summary

## Compare experiments

Generate an experiment comparison report:

```bash
python scripts/compare_experiments.py
```

This writes:

```text
results/experiment_summary.csv
```

and prints a ranked comparison of archived experiments.

## Current experiment status

The project currently includes at least two tracked experiment configurations:

* `linear_fast`
* `linear_long`

So far, `linear_long` performs better, with lower training and evaluation loss and a prediction closer to the target linear relationship.

Example comparison summary:

* `linear_fast`

  * fewer epochs
  * faster run
  * higher loss
  * rougher fit

* `linear_long`

  * more epochs
  * lower loss
  * better fit
  * better prediction accuracy

## MLflow tracking

Log experiment runs to MLflow:

```bash
python scripts/mlflow_run.py configs/experiments/linear_fast.cfg
python scripts/mlflow_run.py configs/experiments/linear_long.cfg
```

Start the MLflow UI:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

In the MLflow UI, look under the `tinyml-ops` experiment to inspect:

* run parameters
* run metrics
* logged artifacts
* archived outputs

## Continuous integration

GitHub Actions workflows are included to validate:

* configure and build
* test execution
* DVC pipeline execution

These workflows live in:

```text
.github/workflows/
```

## Testing

The project currently includes unit and integration coverage for:

* smoke test
* matrix operations
* dense layer forward pass
* activation behavior
* loss computation
* model abstraction
* single-step training
* multi-epoch linear fit
* dataset loading
* config loading
* checkpoint save/load
* evaluation behavior
* prediction behavior

Run tests inside Docker:

```bash
ctest --test-dir build-docker --output-on-failure
```

## Artifacts produced by the workflow

The current workflow can generate:

### Metrics

* `metrics/train_metrics.json`
* `metrics/eval_metrics.json`

### Archived metrics

* `metrics/archive/*`

### Checkpoints

* `models/checkpoints/linear_model.txt`

### Archived checkpoints

* `models/registry/*`

### Reports

* `results/experiment_summary.csv`

## Current limitations

This project is intentionally small in scope for clarity and inspectability.

Current limitations include:

* single-feature regression workflow
* single dense layer training path
* simple CSV parsing
* basic checkpoint format
* local MLflow setup
* local DVC setup
* Docker-first execution path
* no advanced normalization pipeline yet
* no multi-layer experiment workflow yet
* no remote artifact storage yet

These constraints are deliberate so the system remains easy to inspect, debug, and extend.

## Roadmap

Planned next steps include:

* richer dataset handling
* broader model support
* improved checkpoint and artifact handling
* architecture and workflow documentation in `docs/`
* DVC remote storage
* improved experiment management
* stronger model/version promotion workflow
* broader CI and deployment workflow

## Project goals

This project is designed as both:

* a learning-oriented systems implementation in C
* a compact portfolio-grade MLOps workflow

It prioritizes:

* transparency
* reproducibility
* inspectability
* incremental engineering
* practical workflow design

## Getting started summary

If you want the shortest path to run the full system:

```bash
docker build -f docker/Dockerfile.dev -t tinyml-ops-dev .
docker run --rm -it -v ${PWD}:/workspace tinyml-ops-dev
```

Inside the container:

```bash
cmake -S . -B build-docker -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build build-docker
ctest --test-dir build-docker --output-on-failure
python -m dvc repro
python scripts/run_experiment.py configs/experiments/linear_long.cfg
python scripts/compare_experiments.py
```

## Status

The project is currently at a strong early-stage MLOps milestone:

* core ML logic works
* reproducible training works
* evaluation and prediction work
* experiments can be compared
* metrics and checkpoints are tracked
* CI is green
* Docker execution is working
* MLflow tracking is integrated

```

The next best thing to build after this README is `docs/architecture.md` so the workflow is explained visually and structurally.
```