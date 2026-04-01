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
    TinyML_Dataset train_dataset;
    TinyML_Dataset val_dataset;
    TinyML_Dataset test_dataset;

    if (dataset.sample_count == 0) {
        fprintf(stderr, "Failed to load dataset: %s\n", config.data_path);
        return 1;
    }

    tinyml_dataset_split_three_way(
        &dataset,
        config.validation_split,
        config.test_split,
        config.shuffle,
        config.split_seed,
        &train_dataset,
        &val_dataset,
        &test_dataset
    );

    TinyML_NormalizationStats norm_stats;
    if (!tinyml_load_normalization_stats(config.normalization_path, &norm_stats)) {
        fprintf(stderr, "Failed to load normalization stats: %s\n", config.normalization_path);
        tinyml_dataset_free(&dataset);
        tinyml_dataset_free(&train_dataset);
        tinyml_dataset_free(&val_dataset);
        tinyml_dataset_free(&test_dataset);
        return 1;
    }

    tinyml_apply_normalization(&train_dataset, &norm_stats);
    tinyml_apply_normalization(&val_dataset, &norm_stats);
    tinyml_apply_normalization(&test_dataset, &norm_stats);

    TinyML_RuntimeModel model;
    if (!tinyml_runtime_model_load_checkpoint(&model, &config, config.checkpoint_path)) {
        fprintf(stderr, "Failed to load checkpoint: %s\n", config.checkpoint_path);
        tinyml_normalization_stats_free(&norm_stats);
        tinyml_dataset_free(&dataset);
        tinyml_dataset_free(&train_dataset);
        tinyml_dataset_free(&val_dataset);
        tinyml_dataset_free(&test_dataset);
        return 1;
    }

    float eval_loss = tinyml_runtime_model_evaluate(&model, &test_dataset);
    float prediction_x4 = 0.0f;

    printf("Config: %s\n", config_path);
    printf("Dataset: %s\n", config.data_path);
    printf("Checkpoint: %s\n", config.checkpoint_path);
    printf("Model type: %s\n", config.model_type);
    printf("Test loss: %.6f\n", eval_loss);

    if (test_dataset.feature_count == 1) {
        float normalized_x4 = tinyml_normalize_single_value(4.0f, norm_stats.mean[0], norm_stats.std[0]);
        prediction_x4 = tinyml_runtime_model_predict_single(&model, normalized_x4);

        printf("Prediction for x=4.0: %.6f\n", prediction_x4);
        printf("Normalized x=4.0: %.6f\n", normalized_x4);
    } else {
        printf("Sample prediction skipped: dataset has %zu features.\n", test_dataset.feature_count);
    }

    if (!tinyml_write_eval_metrics_json(
            config.eval_metrics_path,
            eval_loss,
            prediction_x4,
            tinyml_runtime_model_parameter_count(&model),
            tinyml_runtime_model_weight_l2_norm(&model),
            tinyml_runtime_model_max_abs_weight(&model),
            tinyml_runtime_model_bias_l2_norm(&model))) {
        fprintf(stderr, "Warning: failed to write eval metrics file.\n");
    }

    tinyml_runtime_model_free(&model);
    tinyml_normalization_stats_free(&norm_stats);
    tinyml_dataset_free(&dataset);
    tinyml_dataset_free(&train_dataset);
    tinyml_dataset_free(&val_dataset);
    tinyml_dataset_free(&test_dataset);

    return 0;
}