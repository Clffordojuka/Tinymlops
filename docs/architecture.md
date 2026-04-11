## `docs/architectures.md`

````markdown id="5fbey0"
# Architectures

## Overview

`tinyml-ops` v1 supports three regression model families built on the same C-based training and evaluation workflow:

- linear models
- single-hidden-layer MLPs
- deep MLPs

These architectures are designed to stay compact, inspectable, and reproducible while still allowing meaningful experimentation across depth, activation, optimizer choice, and weight initialization.

## Design goals

The architecture layer in `tinyml-ops` was built around a few core goals:

- keep the implementation small enough to fully inspect
- support incremental architectural growth from linear to deep models
- expose architecture choices through config files rather than source edits
- make training, evaluation, and prediction work through a shared runtime interface

## 1. Linear model

The linear model is the simplest supported architecture.

It consists of:

- a single dense layer
- no hidden layer
- a direct mapping from input features to one output

This model is useful as:

- a baseline
- a debugging reference
- a sanity-check architecture for the data pipeline

### Example config pattern

```ini id="sxn6e7"
model_type=linear
optimizer=adam
weight_init=zeros
````

## 2. Single-hidden-layer MLP

The MLP adds one hidden layer between input and output.

It supports:

* configurable hidden dimension
* configurable hidden activation
* configurable optimizer
* configurable weight initialization

This is the main intermediate architecture in the project and serves as the default balance between simplicity and expressiveness.

### Example config pattern

```ini id="w5drmr"
model_type=mlp
hidden_dim=8
hidden_activation=tanh
optimizer=adam
weight_init=xavier
```

### Typical use cases

* stronger nonlinear baseline
* activation comparisons
* optimizer comparisons
* weight initialization comparisons

## 3. Deep MLP

The deep MLP extends the MLP design to multiple hidden layers.

It supports:

* configurable hidden layer sizes
* configurable hidden activation
* configurable optimizer
* configurable weight initialization

### Example config pattern

```ini id="lb3pa0"
model_type=deep_mlp
hidden_layers=16,8
hidden_activation=relu
optimizer=adam
weight_init=he
```

### Typical use cases

* architecture sweeps
* depth comparisons
* hidden-layer-size experiments
* activation and initialization interaction studies

## Hidden activations

The project supports the following hidden activation settings:

* `relu`
* `tanh`
* `linear`
* `none`

In practice for v1:

* `relu` is most natural with `he` initialization
* `tanh` is most natural with `xavier` initialization

## Runtime model abstraction

A major architectural feature in v1 is the runtime model abstraction.

This layer allows the CLI applications to handle multiple architecture families through one interface for:

* initialization
* training
* evaluation
* prediction
* checkpoint load/save

This means the application layer does not need to duplicate architecture-specific branching everywhere.

## Weight initialization support

Architectures in v1 support configurable initialization:

* `zeros`
* `xavier`
* `he`

These initialization choices are applied during model creation and tracked through experiment metadata, benchmark reports, and MLflow.

## Optimizer support

Architectures can currently be trained with:

* SGD
* Adam

This makes architecture comparison more meaningful, since the project can now compare not just model family and depth, but also optimization strategy.

## Current recommended default architecture

The promoted v1 default architecture is:

* model type: `mlp`
* hidden dimension: `8`
* hidden activation: `tanh`
* optimizer: `adam`
* weight initialization: `xavier`

This default was selected from the valid benchmarked experiment set and promoted into:

```text id="tiib3v"
configs/experiments/best_default.cfg
```

## Architectural tradeoffs in v1

### Linear

Strengths:

* easiest to debug
* smallest parameter count
* cleanest baseline

Limitations:

* limited nonlinear capacity

### MLP

Strengths:

* stronger nonlinear modeling
* compact and stable
* good benchmark default

Limitations:

* less expressive than deeper networks

### Deep MLP

Strengths:

* more capacity
* more architecture flexibility

Limitations:

* more sensitive to initialization and optimization
* harder to benchmark reliably without stronger experiment controls

## Summary

The architecture design in `tinyml-ops` v1 provides a clean progression from linear regression to deeper configurable neural networks while preserving transparency, reproducibility, and a shared runtime workflow.

````