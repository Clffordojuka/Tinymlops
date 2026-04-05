# Benchmark Report

## Best Overall Experiment

- **Experiment:** deep_mlp_relu_16_8
- **Model Type:** deep_mlp
- **Hidden Layers:** 16,8
- **Hidden Activation:** relu
- **Optimizer:** 
- **Weight Init:** 
- **Eval Loss:** 0.0
- **Train Loss:** 9.934813
- **Val Loss:** 0.0

## Top 5 Experiments

| Rank | Experiment | Model | Hidden Layers | Activation | Optimizer | Weight Init | Eval Loss | Train Loss | Val Loss |
|---:|---|---|---|---|---|---|---:|---:|---:|
| 1 | deep_mlp_relu_16_8 | deep_mlp | 16,8 | relu |  |  | 0.0 | 9.934813 | 0.0 |
| 2 | deeP_mlp_relu_8_4 | deep_mlp | 8,4 | relu |  |  | 0.0 | 9.934813 | 0.0 |
| 3 | deep_mlp_relu | deep_mlp |  | none |  |  | 0.0 | 9.934813 | 0.0 |
| 4 | deep_mlp_tanh_16_8 | deep_mlp | 16,8 | tanh |  |  | 0.0 | 9.934813 | 0.0 |
| 5 | deep_mlp_tanh_8_4 | deep_mlp | 8,4 | tanh |  |  | 0.0 | 9.934813 | 0.0 |

## Best by Model Type

- **deep_mlp:** deep_mlp_relu_16_8 | model=deep_mlp | hidden_layers=16,8 | activation=relu | optimizer= | weight_init= | eval_loss=0.0 | train_loss=9.934813 | val_loss=0.0
- **linear:** linear_fast | model=linear | hidden_layers= | activation=none | optimizer= | weight_init= | eval_loss=0.0 | train_loss=0.000116 | val_loss=0.000118
- **mlp:** mlp_relu | model=mlp | hidden_layers= | activation=relu | optimizer= | weight_init= | eval_loss=0.0 | train_loss=9.934813 | val_loss=0.0
- **unknown:** linear_long | model= | hidden_layers= | activation= | optimizer= | weight_init= | eval_loss=0.0 | train_loss=0.066707 | val_loss=0.085756

## Best by Hidden Activation

- **none:** deep_mlp_relu | model=deep_mlp | hidden_layers= | activation=none | optimizer= | weight_init= | eval_loss=0.0 | train_loss=9.934813 | val_loss=0.0
- **relu:** deep_mlp_relu_16_8 | model=deep_mlp | hidden_layers=16,8 | activation=relu | optimizer= | weight_init= | eval_loss=0.0 | train_loss=9.934813 | val_loss=0.0
- **tanh:** deep_mlp_tanh_16_8 | model=deep_mlp | hidden_layers=16,8 | activation=tanh | optimizer= | weight_init= | eval_loss=0.0 | train_loss=9.934813 | val_loss=0.0

## Best by Optimizer

- **adam:** mlp_tanh_8_adam_xavier | model=mlp | hidden_layers= | activation=tanh | optimizer=adam | weight_init=xavier | eval_loss=0.001151 | train_loss=6.557673 | val_loss=96.456558
- **unknown:** deep_mlp_relu_16_8 | model=deep_mlp | hidden_layers=16,8 | activation=relu | optimizer= | weight_init= | eval_loss=0.0 | train_loss=9.934813 | val_loss=0.0

## Best by Weight Initialization

- **he:** deep_mlp_16_8_adam_he | model=deep_mlp | hidden_layers=16,8 | activation=relu | optimizer=adam | weight_init=he | eval_loss=1.192808 | train_loss=3.40413 | val_loss=37.684464
- **unknown:** deep_mlp_relu_16_8 | model=deep_mlp | hidden_layers=16,8 | activation=relu | optimizer= | weight_init= | eval_loss=0.0 | train_loss=9.934813 | val_loss=0.0
- **xavier:** mlp_tanh_8_adam_xavier | model=mlp | hidden_layers= | activation=tanh | optimizer=adam | weight_init=xavier | eval_loss=0.001151 | train_loss=6.557673 | val_loss=96.456558
- **zeros:** linear_adam | model=linear | hidden_layers= | activation=none | optimizer=adam | weight_init=zeros | eval_loss=44.739685 | train_loss=75.873726 | val_loss=269.770874

## Recommended Default

The current recommended default configuration is **deep_mlp_relu_16_8**, because it achieved the lowest recorded eval loss of **0.0** among the benchmarked runs.

