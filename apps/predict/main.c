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

    TinyML_NormalizationStats norm_stats;
    if (!tinyml_load_normalization_stats(config.normalization_path, &norm_stats)) {
        fprintf(stderr, "Failed to load normalization stats: %s\n", config.normalization_path);
        return 1;
    }

    TinyML_RuntimeModel model;
    if (!tinyml_runtime_model_load_checkpoint(&model, &config, config.checkpoint_path)) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", config.checkpoint_path);
        tinyml_normalization_stats_free(&norm_stats);
        return 1;
    }

    size_t input_dim = 0;
    if (model.kind == TINYML_MODEL_LINEAR) {
        input_dim = model.linear.input_dim;
    } else if (model.kind == TINYML_MODEL_MLP) {
        input_dim = model.mlp.hidden.input_dim;
    } else if (model.kind == TINYML_MODEL_DEEP_MLP) {
        if (model.deep_mlp.num_layers == 0 || model.deep_mlp.layers == NULL) {
            fprintf(stderr, "Loaded deep MLP has no layers.\n");
            tinyml_runtime_model_free(&model);
            tinyml_normalization_stats_free(&norm_stats);
            return 1;
        }
        input_dim = model.deep_mlp.layers[0].input_dim;
    }

    if (norm_stats.feature_count != input_dim) {
        fprintf(stderr,
                "Normalization feature count (%zu) does not match model input_dim (%zu)\n",
                norm_stats.feature_count,
                input_dim);
        tinyml_runtime_model_free(&model);
        tinyml_normalization_stats_free(&norm_stats);
        return 1;
    }

    if (argc != (int)(2 + input_dim)) {
        fprintf(stderr, "Usage: %s <config_path>", argv[0]);
        for (size_t i = 0; i < input_dim; ++i) {
            fprintf(stderr, " <x%zu>", i + 1);
        }
        fprintf(stderr, "\n");
        fprintf(stderr,
                "Model expects %zu feature value(s), but received %d.\n",
                input_dim,
                argc - 2);
        tinyml_runtime_model_free(&model);
        tinyml_normalization_stats_free(&norm_stats);
        return 1;
    }

    TinyML_Matrix input = tinyml_matrix_create(1, input_dim);

    printf("Config: %s\n", config_path);
    printf("Checkpoint: %s\n", config.checkpoint_path);
    printf("Normalization: %s\n", config.normalization_path);
    printf("Model type: %s\n", config.model_type);

    for (size_t i = 0; i < input_dim; ++i) {
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

    TinyML_Matrix prediction = tinyml_runtime_model_forward(&model, &input);
    printf("Predicted y: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));

    tinyml_matrix_free(&prediction);
    tinyml_matrix_free(&input);
    tinyml_runtime_model_free(&model);
    tinyml_normalization_stats_free(&norm_stats);

    return 0;
}