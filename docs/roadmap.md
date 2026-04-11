## `docs/roadmap.md`

```markdown id="qqglz7"
# Roadmap

## Overview

This roadmap describes the direction of `tinyml-ops` after the completion of version 1.

Version 1 established the project as a compact, reproducible regression experimentation system in C with experiment tracking, benchmarking, and best-config promotion.

The next roadmap items are future enhancements, not unfinished version 1 work.

## Version 1 completed

The following major milestones are complete in v1:

- linear, MLP, and deep MLP support
- configurable hidden activations
- configurable optimizers
- configurable weight initialization
- config-driven train/evaluate/predict workflow
- train/validation/test split handling
- normalization and checkpointing
- DVC pipeline support
- MLflow tracking
- experiment comparison
- benchmark reporting
- best-config promotion
- Docker workflow
- CI and tests
- documentation refresh

## Near-term priorities

### 1. Classification support

This is the strongest next technical milestone.

Planned additions:

- sigmoid or softmax output behavior
- classification-oriented loss functions
- accuracy metrics
- classification configs
- classification-specific tests
- classification benchmark workflow

### 2. Richer dataset handling

Current v1 datasets are intentionally compact numeric CSVs.

Future improvements:

- broader CSV support
- richer validation of input files
- better multi-feature dataset ergonomics
- more dataset fixtures for testing

### 3. Benchmark promotion automation refinement

Version 1 already supports promotion of the best valid config.

Future work can improve this further through:

- stronger tie-breaking rules
- promotion summaries in markdown and JSON
- promotion history tracking
- benchmark lineage support

### 4. Stronger experiment integrity checks

Future experiment-quality improvements may include:

- stricter metadata validation
- better suspicious-run detection
- stronger benchmark filtering
- more consistency checks between configs and archived outputs

## Mid-term roadmap

### 5. Remote artifact and pipeline storage

Potential next step:

- remote DVC storage configuration
- cleaner multi-machine reproducibility
- better artifact portability

### 6. Broader workflow automation

Potential additions:

- richer batch experiment scheduling
- auto-generated benchmark dashboards
- release-oriented workflow helpers
- better report generation automation

### 7. Expanded CI coverage

Potential CI improvements:

- more environment validation
- artifact checks in CI
- benchmark/report smoke checks
- stricter release-pipeline validation

## Documentation roadmap

Future documentation improvements may include:

- deeper math notes for forward/backward propagation
- architecture decision records
- experiment design notes
- benchmark interpretation notes
- troubleshooting guides

## Long-term direction

Longer-term, `tinyml-ops` could grow in one of two directions:

### Direction A: educational systems project
Keep the project compact and highly inspectable for learning and demonstration.

### Direction B: broader compact ML platform
Expand it into a stronger experimentation platform with more task types, richer metrics, and stronger workflow automation.

Version 1 intentionally prioritizes Direction A while laying groundwork for selective Direction B features.

## Proposed next milestone

A strong next milestone would be:

**tinyml-ops v2: classification support and broader experiment coverage**

That would make the project meaningfully broader without losing the compact design philosophy.

## Summary

The roadmap after version 1 focuses on extending the project from a completed regression experimentation system into a broader but still inspectable ML workflow platform.
````

If you want, I can also give you a very short “copy this into each file” version with no extra explanation.