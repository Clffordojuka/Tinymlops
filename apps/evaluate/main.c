#include <stdio.h>
#include "tinyml.h"

int main(int argc, char **argv) {
    const char *config_path = "configs/base/train_linear.cfg";
    TinyML_TrainConfig config = tinyml_default_train_config();

    if (argc > 1) {
        config_path = argv[1];
    }

    if (!tinyml_load_train_config(config_path, &config)) {
        fprintf(stderr, "Failed to load config: %s\n", config_path);
        return 1;
    }

    TinyML_Dataset dataset = tinyml_dataset_load_csv(config.data_path);
    TinyML_DenseLayer layer;

    if (dataset.sample_count == 0) {
        fprintf(stderr, "Failed to load dataset: %s\n", config.data_path);
        return 1;
    }

    if (!tinyml_load_dense_checkpoint(config.checkpoint_path, &layer)) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", config.checkpoint_path);
        tinyml_dataset_free(&dataset);
        return 1;
    }

    float eval_loss = tinyml_evaluate_dense(&layer, &dataset);

    TinyML_Matrix test_input = tinyml_matrix_create(1, 1);
    tinyml_matrix_set(&test_input, 0, 0, 4.0f);
    TinyML_Matrix prediction = tinyml_dense_forward(&layer, &test_input);

    printf("Config: %s\n", config_path);
    printf("Dataset: %s\n", config.data_path);
    printf("Checkpoint: %s\n", config.checkpoint_path);
    printf("Evaluation loss: %.6f\n", eval_loss);
    printf("Prediction for x=4.0: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));
    printf("Loaded weight: %.6f\n", tinyml_matrix_get(&layer.weights, 0, 0));
    printf("Loaded bias: %.6f\n", tinyml_matrix_get(&layer.bias, 0, 0));

    tinyml_matrix_free(&prediction);
    tinyml_matrix_free(&test_input);
    tinyml_dataset_free(&dataset);
    tinyml_dense_free(&layer);

    return 0;
}