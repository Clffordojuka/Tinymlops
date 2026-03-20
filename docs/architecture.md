# tinyml-ops Architecture

## Purpose

`tinyml-ops` is a compact machine learning systems project that combines a small C-based learning core with reproducible workflow tooling.

It is designed to show how a low-level ML implementation can be wrapped in practical MLOps structure without depending on large training frameworks.

The project focuses on:

- transparency
- reproducibility
- experiment tracking
- workflow automation
- incremental system design

## System overview

At a high level, the project has two layers:

### 1. ML core layer
The C code implements the learning system itself.

This layer includes:

- matrix operations
- dense layer forward pass
- dense layer backward pass
- ReLU activation
- MSE loss
- SGD-style weight updates
- checkpoint save/load
- CSV data loading

### 2. Workflow layer
The surrounding tooling turns the ML core into a reproducible experiment workflow.

This layer includes:

- config files
- train / evaluate / predict applications
- DVC pipeline orchestration
- experiment runner scripts
- experiment comparison reporting
- MLflow tracking
- Docker execution
- GitHub Actions CI

## Core components

### C library
The reusable ML core lives under:

```text
include/
src/
````

Key responsibilities:

* represent matrices and simple models
* run forward and backward passes
* update parameters
* load datasets
* save and load checkpoints
* compute losses and evaluation metrics

### Applications

#### `train_app`

Responsibilities:

* load training config
* load dataset
* train model
* write checkpoint
* write training metrics

#### `evaluate_app`

Responsibilities:

* load config
* load dataset
* load checkpoint
* compute evaluation loss
* write evaluation metrics

#### `predict_app`

Responsibilities:

* load config
* load checkpoint
* run single-value inference
* print prediction results

## Workflow architecture

The system currently follows this flow:

```text
config file
   ↓
train_app
   ↓
checkpoint + train_metrics.json
   ↓
evaluate_app
   ↓
eval_metrics.json
   ↓
run_experiment.py
   ↓
archived metrics + archived checkpoint
   ↓
compare_experiments.py
   ↓
experiment_summary.csv
   ↓
mlflow_run.py
   ↓
MLflow experiment tracking
```

## Artifact flow

### Input artifacts

* config files
* CSV dataset
* DVC params

### Intermediate artifacts

* active checkpoint
* active training metrics
* active evaluation metrics

### Archived artifacts

* archived train metrics
* archived eval metrics
* archived model checkpoints

### Report artifacts

* experiment summary CSV
* MLflow logged artifacts

## DVC pipeline role

DVC is responsible for reproducible execution of the main workflow stages.

The pipeline currently orchestrates:

* training
* evaluation

DVC tracks:

* params
* stage dependencies
* stage outputs
* metrics files
* lock state

This makes it possible to rerun the workflow in a controlled and repeatable way.

## MLflow role

MLflow sits on top of the reproducible workflow and provides experiment tracking.

It records:

* experiment name
* config path
* training parameters
* train and eval metrics
* archived metric files
* archived checkpoint artifacts

MLflow is used here as the experiment history and inspection layer, while DVC remains the reproducibility layer.

## Docker role

Docker is the recommended execution environment.

It solves the main portability issue encountered during setup:

* native Windows GCC builds were inconsistent
* Linux container builds were stable

Docker is therefore treated as the canonical runtime for:

* build
* test
* DVC repro
* experiment execution
* experiment comparison

## GitHub Actions role

GitHub Actions provides automated validation for the repository.

The workflows currently verify:

* project configure/build
* test suite execution
* DVC pipeline execution

This protects the repo from regressions and keeps the workflow reproducible in CI.

## Current experiment model

The current project supports a simple regression workflow based on:

* one input feature
* one output target
* one dense layer
* MSE loss
* simple gradient descent updates

This makes the system intentionally small and easy to inspect.

## Current strengths

The project already demonstrates:

* low-level model implementation in C
* config-driven workflows
* reproducible pipelines
* archived experiment history
* experiment comparison
* tracked metrics and artifacts
* CI validation
* Docker portability

## Current limitations

The system is still intentionally narrow in scope.

Known limitations include:

* single-feature regression only
* single-layer training path
* simple checkpoint format
* simple CSV parser
* no batching abstraction
* no normalization pipeline
* no richer model registry workflow
* no remote artifact storage yet

## Extension points

The architecture is intentionally modular so future work can extend it in several directions.

### ML core extensions

* deeper networks
* additional activations
* additional losses
* optimizers beyond simple SGD
* normalization support

### Data extensions

* multi-column datasets
* train/validation/test split support
* preprocessing pipeline
* feature scaling and persistence

### Workflow extensions

* richer experiment catalog
* remote DVC storage
* better model promotion flow
* deployment-oriented packaging
* richer reporting

## Design philosophy

The project deliberately avoids large abstractions too early.

Instead, it follows a staged approach:

1. make the core logic work
2. test it
3. make it reproducible
4. make experiments comparable
5. make the workflow portable
6. layer in tracking and CI

This keeps the system understandable while still showing real MLOps practices.

## Current status

At the current milestone, `tinyml-ops` is a functioning compact ML systems project with:

* a working C learning core
* reproducible DVC pipelines
* experiment archiving
* comparison reporting
* MLflow integration
* Docker-based execution
* CI validation

```