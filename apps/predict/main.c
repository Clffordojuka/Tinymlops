#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

    TinyML_NormalizationStats norm_stats;
    if (!tinyml_load_normalization_stats(config.normalization_path, &norm_stats)) {
        fprintf(stderr, "Failed to load normalization stats: %s\n", config.normalization_path);
        return 1;
    }

    printf("Config: %s\n", config_path);
    printf("Checkpoint: %s\n", config.checkpoint_path);
    printf("Normalization: %s\n", config.normalization_path);

    if (strcmp(config.model_type, "mlp") == 0) {
        TinyML_MLP mlp;
        if (!tinyml_load_mlp_checkpoint(config.checkpoint_path, &mlp)) {
            fprintf(stderr, "Failed to load MLP checkpoint: %s\n", config.checkpoint_path);
            tinyml_normalization_stats_free(&norm_stats);
            return 1;
        }

        if (norm_stats.feature_count != mlp.hidden.input_dim) {
            fprintf(stderr,
                    "Normalization feature count (%zu) does not match model input_dim (%zu)\n",
                    norm_stats.feature_count,
                    mlp.hidden.input_dim);
            tinyml_normalization_stats_free(&norm_stats);
            tinyml_mlp_free(&mlp);
            return 1;
        }

        if (argc != (int)(2 + mlp.hidden.input_dim)) {
            fprintf(stderr, "Usage: %s <config_path>", argv[0]);
            for (size_t i = 0; i < mlp.hidden.input_dim; ++i) {
                fprintf(stderr, " <x%zu>", i + 1);
            }
            fprintf(stderr, "\n");
            tinyml_normalization_stats_free(&norm_stats);
            tinyml_mlp_free(&mlp);
            return 1;
        }

        TinyML_Matrix input = tinyml_matrix_create(1, mlp.hidden.input_dim);

        for (size_t i = 0; i < mlp.hidden.input_dim; ++i) {
            float raw_value = (float)atof(argv[2 + i]);
            float normalized_value = tinyml_normalize_single_value(raw_value, norm_stats.mean[i], norm_stats.std[i]);
            tinyml_matrix_set(&input, 0, i, normalized_value);

            printf("Input x%zu: %.6f\n", i + 1, raw_value);
            printf("Normalized x%zu: %.6f\n", i + 1, normalized_value);
        }

        TinyML_Matrix prediction = tinyml_mlp_forward(&mlp, &input);
        printf("Model type: mlp\n");
        printf("Predicted y: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));

        tinyml_matrix_free(&prediction);
        tinyml_matrix_free(&input);
        tinyml_mlp_free(&mlp);
    } else {
        TinyML_DenseLayer layer;
        if (!tinyml_load_dense_checkpoint(config.checkpoint_path, &layer)) {
            fprintf(stderr, "Failed to load checkpoint: %s\n", config.checkpoint_path);
            tinyml_normalization_stats_free(&norm_stats);
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
            tinyml_normalization_stats_free(&norm_stats);
            tinyml_dense_free(&layer);
            return 1;
        }

        TinyML_Matrix input = tinyml_matrix_create(1, layer.input_dim);

        for (size_t i = 0; i < layer.input_dim; ++i) {
            float raw_value = (float)atof(argv[2 + i]);
            float normalized_value = tinyml_normalize_single_value(raw_value, norm_stats.mean[i], norm_stats.std[i]);
            tinyml_matrix_set(&input, 0, i, normalized_value);

            printf("Input x%zu: %.6f\n", i + 1, raw_value);
            printf("Normalized x%zu: %.6f\n", i + 1, normalized_value);
        }

        TinyML_Matrix prediction = tinyml_dense_forward(&layer, &input);
        printf("Model type: linear\n");
        printf("Predicted y: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));

        tinyml_matrix_free(&prediction);
        tinyml_matrix_free(&input);
        tinyml_dense_free(&layer);
    }

    tinyml_normalization_stats_free(&norm_stats);
    return 0;
}