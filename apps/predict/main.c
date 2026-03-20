#include <stdio.h>
#include <stdlib.h>
#include "tinyml.h"

int main(int argc, char **argv) {
    const char *config_path = "configs/base/train_linear.cfg";
    float x_value = 4.0f;
    TinyML_TrainConfig config = tinyml_default_train_config();

    if (argc > 1) {
        config_path = argv[1];
    }

    if (argc > 2) {
        x_value = (float)atof(argv[2]);
    }

    if (!tinyml_load_train_config(config_path, &config)) {
        fprintf(stderr, "Failed to load config: %s\n", config_path);
        return 1;
    }

    TinyML_DenseLayer layer;
    TinyML_NormalizationStats norm_stats;

    if (!tinyml_load_dense_checkpoint(config.checkpoint_path, &layer)) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", config.checkpoint_path);
        return 1;
    }

    if (!tinyml_load_normalization_stats(config.normalization_path, &norm_stats)) {
        fprintf(stderr, "Failed to load normalization stats: %s\n", config.normalization_path);
        tinyml_dense_free(&layer);
        return 1;
    }

    float normalized_x = tinyml_normalize_single_value(x_value, norm_stats.mean[0], norm_stats.std[0]);
    float prediction = tinyml_predict_dense_single(&layer, normalized_x);

    printf("Config: %s\n", config_path);
    printf("Checkpoint: %s\n", config.checkpoint_path);
    printf("Normalization: %s\n", config.normalization_path);
    printf("Input x: %.6f\n", x_value);
    printf("Normalized x: %.6f\n", normalized_x);
    printf("Predicted y: %.6f\n", prediction);
    printf("Model weight: %.6f\n", tinyml_matrix_get(&layer.weights, 0, 0));
    printf("Model bias: %.6f\n", tinyml_matrix_get(&layer.bias, 0, 0));

    tinyml_normalization_stats_free(&norm_stats);
    tinyml_dense_free(&layer);
    return 0;
}