## `docs/release-process.md`

```markdown id="4e4xej"
# Release Process

## Overview

This document describes the practical release process used for `tinyml-ops` v1.

The goal of the release process is to make sure a project milestone is:

- built successfully
- tested successfully
- benchmarked with valid results
- documented clearly
- promoted with a recommended default config

## Version 1 release criteria

Version 1 is considered complete when the following are true:

- the project builds successfully in Docker
- all tests pass
- train/evaluate/predict workflows work
- experiment comparison works
- benchmark reporting works
- best-config promotion works
- README and docs are updated
- final workflow artifacts are present

## Recommended release environment

Releases should be validated inside Docker to reduce environment inconsistency.

### Build image

```bash id="wxmjlwm"
docker build -f docker/Dockerfile.dev -t tinyml-ops-dev .
````

### Start container

```bash id="p50owd"
docker run --rm -it -v ${PWD}:/workspace tinyml-ops-dev
```

## Release checklist

### 1. Configure and build

```bash id="22czlx"
cmake -S . -B build-docker -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build build-docker
```

### 2. Run all tests

```bash id="91lmcw"
ctest --test-dir build-docker --output-on-failure
```

### 3. Run the reproducible pipeline

```bash id="313hku"
python -m dvc repro
```

### 4. Run batch experiments

```bash id="jbv4vg"
python scripts/run_experiment_batch.py configs/experiments --mlflow
```

This should complete:

* experiment execution
* comparison generation
* benchmark report generation
* best-config promotion
* batch summary generation

### 5. Verify release artifacts

Confirm these exist:

```text id="5vaehn"
results/experiment_summary.csv
results/benchmark_report.md
results/batch_summary.json
results/promoted_default.json
configs/experiments/best_default.cfg
```

### 6. Verify promoted config

Run the promoted default config directly:

```bash id="sglq77"
python scripts/run_experiment.py configs/experiments/best_default.cfg
```

Optional direct app check:

```bash id="rd4k9f"
./build-docker/apps/train_app configs/experiments/best_default.cfg
./build-docker/apps/evaluate_app configs/experiments/best_default.cfg
```

### 7. Review documentation

Ensure the following are current:

* `README.md`
* `docs/architectures.md`
* `docs/experiments.md`
* `docs/release-process.md`
* `docs/roadmap.md`

## Release outputs

A completed release should include:

### Core outputs

* successful build
* successful tests
* reproducible pipeline run

### Metrics outputs

* training metrics
* evaluation metrics
* archived metrics

### Model outputs

* active checkpoint
* archived checkpoints
* exported parameter summaries

### Reporting outputs

* experiment comparison CSV
* benchmark report
* promoted default metadata
* batch summary

## Promotion policy

The release should not promote a default config manually unless necessary.

The preferred v1 policy is:

1. generate experiment comparison data
2. generate benchmark report using valid-run filtering
3. promote the best valid configuration automatically
4. record promotion metadata

This ensures the release default is evidence-based rather than chosen informally.

## Failure handling

A release should be paused if any of the following occur:

* build failure
* test failure
* broken DVC pipeline
* missing benchmark report
* promotion failure
* missing final artifacts
* invalid benchmark winner due to incomplete metadata

## Version naming

A suitable label for this milestone is:

**tinyml-ops v1: regression experimentation and MLOps workflow complete**

## Post-release direction

After v1 release, future work should be treated as v2 expansion, not unfinished v1 work.

Examples of post-v1 work:

* classification support
* richer dataset handling
* stronger deployment workflows
* remote DVC storage
* richer experiment automation

## Summary

The v1 release process for `tinyml-ops` is complete when the project is fully built, tested, benchmarked, documented, and promoted with a valid default config.

````