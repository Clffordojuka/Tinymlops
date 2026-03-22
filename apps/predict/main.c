#include <stdio.h>
#include <stdlib.h>
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

    if (norm_stats.feature_count != layer.input_dim) {
        fprintf(stderr,
                "Normalization feature count (%zu) does not match model input_dim (%zu)\n",
                norm_stats.feature_count,
                layer.input_dim);
        tinyml_normalization_stats_free(&norm_stats);
        tinyml_dense_free(&layer);
        return 1;
    }

    if (argc != (int)(2 + layer.input_dim)) {
        fprintf(stderr, "Usage: %s <config_path>", argv[0]);
        for (size_t i = 0; i < layer.input_dim; ++i) {
            fprintf(stderr, " <x%zu>", i + 1);
        }
        fprintf(stderr, "\n");
        fprintf(stderr,
                "Model expects %zu feature value(s), but received %d.\n",
                layer.input_dim,
                argc - 2);
        tinyml_normalization_stats_free(&norm_stats);
        tinyml_dense_free(&layer);
        return 1;
    }

    TinyML_Matrix input = tinyml_matrix_create(1, layer.input_dim);

    printf("Config: %s\n", config_path);
    printf("Checkpoint: %s\n", config.checkpoint_path);
    printf("Normalization: %s\n", config.normalization_path);

    for (size_t i = 0; i < layer.input_dim; ++i) {
        float raw_value = (float)atof(argv[2 + i]);
        float normalized_value = tinyml_normalize_single_value(
            raw_value,
            norm_stats.mean[i],
            norm_stats.std[i]
        );

        tinyml_matrix_set(&input, 0, i, normalized_value);

        printf("Input x%zu: %.6f\n", i + 1, raw_value);
        printf("Normalized x%zu: %.6f\n", i + 1, normalized_value);
    }

    TinyML_Matrix prediction = tinyml_dense_forward(&layer, &input);

    printf("Predicted y: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));

    for (size_t i = 0; i < layer.input_dim; ++i) {
        for (size_t o = 0; o < layer.output_dim; ++o) {
            printf("Model weight[%zu,%zu]: %.6f\n",
                   i, o, tinyml_matrix_get(&layer.weights, i, o));
        }
    }

    for (size_t o = 0; o < layer.output_dim; ++o) {
        printf("Model bias[%zu]: %.6f\n", o, tinyml_matrix_get(&layer.bias, 0, o));
    }

    tinyml_matrix_free(&prediction);
    tinyml_matrix_free(&input);
    tinyml_normalization_stats_free(&norm_stats);
    tinyml_dense_free(&layer);

    return 0;
}