## `docs/experiments.md`

```markdown id="q0vi8g"
# Experiments

## Overview

`tinyml-ops` v1 supports reproducible, config-driven regression experiments across model architecture, activation, optimizer, and initialization choices.

Experiments are intended to answer questions such as:

- which model family performs best?
- which hidden activation works best?
- which optimizer is strongest?
- which weight initialization is most stable?
- which configuration should become the recommended default?

## Experiment structure

Each experiment is defined by a `.cfg` file.

Configs can specify:

- dataset path
- model type
- hidden dimension
- hidden layer list
- hidden activation
- optimizer
- Adam hyperparameters
- learning-rate schedule
- L2 regularization
- weight initialization
- batch size
- split ratios
- output paths

This design allows experiments to be changed through config edits rather than code rewrites.

## Core experiment dimensions in v1

### Model families
- linear
- MLP
- deep MLP

### Hidden activations
- relu
- tanh
- linear
- none

### Optimizers
- SGD
- Adam

### Weight initialization
- zeros
- xavier
- he

## Single experiment execution

A single experiment can be run with:

```bash id="wg3mns"
python scripts/run_experiment.py configs/experiments/best_default.cfg
````

This process:

1. updates `params.yaml`
2. runs the DVC pipeline
3. collects train and eval metrics
4. exports model parameters
5. archives outputs
6. prints a summary

## Batch experiment execution

A sweep of configs can be run with:

```bash id="9p0v42"
python scripts/run_experiment_batch.py configs/experiments
```

or with MLflow tracking:

```bash id="bkg5up"
python scripts/run_experiment_batch.py configs/experiments --mlflow
```

The batch runner performs:

* config discovery
* sequential experiment execution
* optional MLflow logging
* comparison CSV generation
* benchmark report generation
* best-config promotion
* batch summary generation

## Comparison workflow

Experiment results are compared using:

```bash id="qjlwm8"
python scripts/compare_experiments.py
```

This produces:

```text id="jlwmwe"
results/experiment_summary.csv
```

The CSV becomes the ranked source for later benchmark reporting.

## Benchmark report

Benchmarking is performed with:

```bash id="o9wxqv"
python scripts/report_best_experiments.py
```

This produces:

```text id="r6d3mp"
results/benchmark_report.md
```

The benchmark report includes:

* total rows found
* valid comparable runs
* excluded runs
* best overall valid experiment
* top ranked valid experiments
* best by model type
* best by activation
* best by optimizer
* best by weight initialization

## Validity filtering

Benchmark integrity was an important v1 improvement.

The report excludes runs with problems such as:

* missing model type
* missing optimizer
* missing weight initialization
* missing validation loss
* missing hidden activation for MLP/deep MLP
* suspicious zero evaluation and validation loss
* malformed or legacy experiment names

This makes benchmark conclusions safer and more meaningful.

## Best-config promotion

After benchmarking, the best valid experiment can be promoted automatically into:

```text id="mjlwm2"
configs/experiments/best_default.cfg
```

Promotion metadata is stored in:

```text id="ojy39z"
results/promoted_default.json
```

This turns the benchmark workflow into an actual configuration-management step rather than just a report.

## Current promoted default

At the end of v1, the project supports a benchmark-backed promoted default config intended for:

* demos
* onboarding
* standard experiments
* README examples
* future regression baselines

## MLflow support

Experiments can also be logged to MLflow with:

```bash id="ohm9c3"
python scripts/mlflow_run.py <config_path>
```

MLflow captures:

* architecture metadata
* optimizer metadata
* initialization metadata
* training metrics
* evaluation metrics
* artifacts

## Experiment artifacts

Experiments may produce:

* `metrics/train_metrics.json`
* `metrics/eval_metrics.json`
* archived metrics in `metrics/archive/`
* checkpoints in `models/checkpoints/`
* archived checkpoints in `models/registry/`
* exported parameters in `models/exported/`
* comparison CSV in `results/`
* benchmark reports in `results/`
* promoted default metadata in `results/`

## Summary

The experiment system in `tinyml-ops` v1 turns low-level C model training into a reproducible benchmarking workflow with ranking, reporting, and promotion of the best valid configuration.

````