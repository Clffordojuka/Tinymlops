# tinyml-ops

A C-first machine learning systems project that combines a lightweight neural-network core with reproducible training, experiment tracking, automated testing, and workflow-oriented MLOps practices.

## Overview

`tinyml-ops` started as a small neural network implementation in C and has grown into a compact end-to-end ML systems project focused on transparency, reproducibility, and inspectable workflows.

The project now supports:

- matrix operations
- dense layer forward and backward propagation
- configurable hidden activations (`relu`, `tanh`, `linear`, `none`)
- MSE loss
- linear models
- single-hidden-layer MLPs
- deep MLPs with configurable hidden layer sizes
- mini-batch training
- learning-rate scheduling
- L2 regularization
- early stopping
- CSV dataset loading
- config-driven training and evaluation
- train/validation/test dataset splitting
- normalization fit/save/load
- checkpoint save/load for linear, MLP, and deep MLP models
- runtime model abstraction for train/evaluate/predict
- DVC-based reproducible pipelines
- experiment archiving and comparison
- MLflow tracking
- automated batch experiment execution
- unit and integration testing
- Docker-based reproducible execution
- GitHub Actions CI

## Repository structure

```text
tinyml-ops/
├── include/
│   └── tinyml.h
├── src/
│   ├── core/
│   ├── layers/
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

## Supported model types

The framework currently supports three model families.

### 1. Linear

A single dense layer with no hidden layer.

Example:

```ini
model_type=linear
```

### 2. MLP

A single-hidden-layer neural network.

Example:

```ini
model_type=mlp
hidden_dim=8
hidden_activation=relu
```

### 3. Deep MLP

A multi-hidden-layer neural network with configurable hidden layer sizes.

Example:

```ini
model_type=deep_mlp
hidden_layers=16,8,4
hidden_activation=tanh
```

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

The project provides three CLI applications:

* `train_app`
* `evaluate_app`
* `predict_app`

### Train

Trains a model from a config file and produces:

* checkpoint
* training metrics
* normalization statistics
* console training logs

### Evaluate

Loads a checkpoint and dataset, then reports:

* evaluation loss
* sample prediction
* model architecture-aware metrics

### Predict

Loads a checkpoint and predicts a value for one input sample.

## Configuration

Training and evaluation are controlled through config files.

### Base config

```text
configs/base/train_linear.cfg
```

### Example experiment configs

```text
configs/experiments/linear_fast.cfg
configs/experiments/linear_long.cfg
configs/experiments/mlp_relu_8.cfg
configs/experiments/mlp_tanh_8.cfg
configs/experiments/deep_mlp_8_4.cfg
configs/experiments/deep_mlp_tanh_8_4.cfg
```

Configs define values such as:

* dataset path
* epoch count
* learning rate
* learning-rate schedule
* L2 regularization
* model type
* hidden dimension or hidden layer list
* hidden activation
* batch size
* validation split
* test split
* metrics output path
* checkpoint output path
* normalization output path

## Example config patterns

### Linear

```ini
model_type=linear
learning_rate=0.01
epochs=200
```

### Single-hidden-layer MLP

```ini
model_type=mlp
hidden_dim=8
hidden_activation=relu
learning_rate=0.005
epochs=500
```

### Deep MLP

```ini
model_type=deep_mlp
hidden_layers=8,4
hidden_activation=relu
learning_rate=0.005
epochs=500
```

## Training

Train from a config file inside Docker:

```bash
./build-docker/apps/train_app configs/experiments/linear_long.cfg
```

or:

```bash
./build-docker/apps/train_app configs/experiments/deep_mlp_8_4.cfg
```

Typical output includes:

* epoch-by-epoch training and validation loss
* learning-rate updates
* best epoch
* early stopping status
* sample prediction for a reference input
* model type and architecture summary

## Evaluation

Evaluate a trained checkpoint:

```bash
./build-docker/apps/evaluate_app configs/experiments/deep_mlp_8_4.cfg
```

Typical output includes:

* test loss
* sample prediction
* model-type-aware evaluation metrics

## Prediction

Run inference for a single input sample:

```bash
./build-docker/apps/predict_app configs/experiments/deep_mlp_8_4.cfg 4.0
```

For multi-feature models:

```bash
./build-docker/apps/predict_app <config_path> <x1> <x2> ...
```

Example output includes:

* input values
* normalized inputs
* predicted output
* checkpoint path
* model type

## Dataset handling

The current project uses compact CSV regression datasets.

Example sample file:

```text
data/samples/linear.csv
```

Current dataset support includes:

* numeric CSV parsing
* train/validation/test splitting
* deterministic shuffling with seed
* normalization fit on training split only
* normalization reuse for evaluation and prediction

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
* normalization output
* training metrics
* evaluation metrics

## Experiment runner

Use the experiment runner to automate config switching, DVC reproduction, export, and archiving:

```bash
python scripts/run_experiment.py configs/experiments/linear_long.cfg
python scripts/run_experiment.py configs/experiments/deep_mlp_8_4.cfg
```

This script:

* updates `params.yaml`
* runs `dvc repro`
* reads train and eval metrics
* exports model parameters
* archives train metrics into `metrics/archive/`
* archives eval metrics into `metrics/archive/`
* archives checkpoints into `models/registry/`
* archives exported model parameters into `models/exported/`
* prints a concise summary

## Batch experiment runner

Run multiple experiment configs in one command:

```bash
python scripts/run_experiment_batch.py configs/experiments
```

With MLflow logging enabled:

```bash
python scripts/run_experiment_batch.py configs/experiments --mlflow
```

The batch runner:

* discovers `.cfg` files
* runs experiments sequentially
* optionally logs runs to MLflow
* updates experiment comparison outputs
* writes a batch summary

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

## MLflow tracking

Log experiment runs to MLflow:

```bash
python scripts/mlflow_run.py configs/experiments/linear_long.cfg
python scripts/mlflow_run.py configs/experiments/deep_mlp_8_4.cfg
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
* model architecture metadata
* logged artifacts
* archived outputs

## Testing

The project includes unit and integration coverage for:

* smoke tests
* matrix operations
* dense layer forward pass
* activation behavior
* loss computation
* model abstraction
* single-step training
* multi-epoch fitting
* dataset loading
* config loading
* hidden-layer parsing
* checkpoint save/load
* MLP checkpoint save/load
* deep MLP checkpoint save/load
* evaluation behavior
* prediction behavior
* deep MLP forward behavior
* deep MLP training-step behavior

Run tests inside Docker:

```bash
ctest --test-dir build-docker --output-on-failure
```

## Artifacts produced by the workflow

### Metrics

* `metrics/train_metrics.json`
* `metrics/eval_metrics.json`

### Archived metrics

* `metrics/archive/*`

### Checkpoints

* `models/checkpoints/linear_model.txt`

### Archived checkpoints

* `models/registry/*`

### Exported model parameters

* `models/exported/model_params.json`
* `models/exported/*`

### Reports

* `results/experiment_summary.csv`
* `results/batch_summary.json`

## Continuous integration

GitHub Actions workflows are included to validate:

* configure and build
* test execution
* pipeline execution

These workflows live in:

```text
.github/workflows/
```

## Current limitations

This project is still intentionally compact in scope for clarity and inspectability.

Current limitations include:

* regression-oriented workflow
* MSE-loss-focused training path
* simple CSV parsing
* local MLflow setup
* local DVC setup
* Docker-first execution path
* no optimizer choices beyond the current training implementation
* no classification losses yet
* no remote artifact storage yet

These constraints are deliberate so the system remains easy to inspect, debug, and extend.

## Roadmap

Planned next steps include:

* optimizer upgrades such as Adam
* improved weight initialization
* richer dataset handling
* classification support
* remote DVC storage
* stronger experiment management
* improved reporting and architecture comparison
* broader CI and deployment workflows
* more detailed design documentation in `docs/`

## Project goals

This project is designed as both:

* a learning-oriented ML systems implementation in C
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
python scripts/run_experiment_batch.py configs/experiments --mlflow
python scripts/compare_experiments.py
```

## Status

The project is currently at a strong compact-ML-systems milestone:

* core ML logic works
* linear, MLP, and deep MLP models work
* reproducible training works
* evaluation and prediction work
* experiments can be compared
* architecture metadata is tracked
* metrics and checkpoints are archived
* CI is working
* Docker execution is working
* MLflow tracking is integrated
* batch experiment execution is working

```