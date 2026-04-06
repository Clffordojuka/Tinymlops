# Benchmark Report

## Benchmark Set Quality

- **Total rows found:** 22
- **Valid comparable runs:** 9
- **Excluded runs:** 13

## Best Overall Experiment

- **Experiment:** best_default
- **Model Type:** mlp
- **Hidden Layers:** 
- **Hidden Activation:** tanh
- **Optimizer:** adam
- **Weight Init:** xavier
- **Eval Loss:** 0.001151
- **Train Loss:** 6.557673
- **Val Loss:** 96.456558

## Top 5 Valid Experiments

| Rank | Experiment | Model | Hidden Layers | Activation | Optimizer | Weight Init | Eval Loss | Train Loss | Val Loss |
|---:|---|---|---|---|---|---|---:|---:|---:|
| 1 | best_default | mlp |  | tanh | adam | xavier | 0.001151 | 6.557673 | 96.456558 |
| 2 | mlp_tanh_8_adam_xavier | mlp |  | tanh | adam | xavier | 0.001151 | 6.557673 | 96.456558 |
| 3 | deep_mlp_tanh_8_4_adam_xavier | deep_mlp | 8,4 | tanh | adam | xavier | 0.426023 | 18.188829 | 166.9039 |
| 4 | mlp_relu_8_adam_xavier | mlp |  | relu | adam | xavier | 0.981091 | 1.329567 | 6.64608 |
| 5 | deep_mlp_16_8_adam_he | deep_mlp | 16,8 | relu | adam | he | 1.192808 | 3.40413 | 37.684464 |

## Best by Model Type

- **deep_mlp:** deep_mlp_tanh_8_4_adam_xavier | model=deep_mlp | hidden_layers=8,4 | activation=tanh | optimizer=adam | weight_init=xavier | eval_loss=0.426023 | train_loss=18.188829 | val_loss=166.9039
- **linear:** linear_adam | model=linear | hidden_layers= | activation=none | optimizer=adam | weight_init=zeros | eval_loss=44.739685 | train_loss=75.873726 | val_loss=269.770874
- **mlp:** best_default | model=mlp | hidden_layers= | activation=tanh | optimizer=adam | weight_init=xavier | eval_loss=0.001151 | train_loss=6.557673 | val_loss=96.456558

## Best by Hidden Activation

- **none:** linear_adam | model=linear | hidden_layers= | activation=none | optimizer=adam | weight_init=zeros | eval_loss=44.739685 | train_loss=75.873726 | val_loss=269.770874
- **relu:** mlp_relu_8_adam_xavier | model=mlp | hidden_layers= | activation=relu | optimizer=adam | weight_init=xavier | eval_loss=0.981091 | train_loss=1.329567 | val_loss=6.64608
- **tanh:** best_default | model=mlp | hidden_layers= | activation=tanh | optimizer=adam | weight_init=xavier | eval_loss=0.001151 | train_loss=6.557673 | val_loss=96.456558

## Best by Optimizer

- **adam:** best_default | model=mlp | hidden_layers= | activation=tanh | optimizer=adam | weight_init=xavier | eval_loss=0.001151 | train_loss=6.557673 | val_loss=96.456558

## Best by Weight Initialization

- **he:** deep_mlp_16_8_adam_he | model=deep_mlp | hidden_layers=16,8 | activation=relu | optimizer=adam | weight_init=he | eval_loss=1.192808 | train_loss=3.40413 | val_loss=37.684464
- **xavier:** best_default | model=mlp | hidden_layers= | activation=tanh | optimizer=adam | weight_init=xavier | eval_loss=0.001151 | train_loss=6.557673 | val_loss=96.456558
- **zeros:** linear_adam | model=linear | hidden_layers= | activation=none | optimizer=adam | weight_init=zeros | eval_loss=44.739685 | train_loss=75.873726 | val_loss=269.770874

## Excluded Runs

| Experiment | Reason | Model | Optimizer | Weight Init | Eval Loss | Val Loss |
|---|---|---|---|---|---:|---:|
| linear_long | missing_model_type |  |  |  | 0.0 | 0.085756 |
| mlp_long | missing_model_type |  |  |  | 0.0 | 0.0 |
| deeP_mlp_relu_8_4 | missing_optimizer | deep_mlp |  |  | 0.0 | 0.0 |
| deep_mlp_relu | missing_optimizer | deep_mlp |  |  | 0.0 | 0.0 |
| deep_mlp_relu_16_8 | missing_optimizer | deep_mlp |  |  | 0.0 | 0.0 |
| deep_mlp_tanh_16_8 | missing_optimizer | deep_mlp |  |  | 0.0 | 0.0 |
| deep_mlp_tanh_8_4 | missing_optimizer | deep_mlp |  |  | 0.0 | 0.0 |
| linear_fast | missing_optimizer | linear |  |  | 0.0 | 0.000118 |
| linear_multi | missing_optimizer | linear |  |  | 6.3e-05 | 2.3e-05 |
| linear_slow | missing_optimizer | linear |  |  | 0.0 | 0.0 |
| mlp_multi | missing_optimizer | mlp |  |  | 3.304271 | 125.042374 |
| mlp_relu | missing_optimizer | mlp |  |  | 0.0 | 0.0 |
| mlp_tanh | missing_optimizer | mlp |  |  | 0.0 | 0.0 |

## Recommended Default

The current recommended default configuration is **best_default**, because it is the best result among the **valid comparable runs** and achieved an eval loss of **0.001151**.

