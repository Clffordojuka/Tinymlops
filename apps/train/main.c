#include <stdio.h>
#include "tinyml.h"

static float tinyml_compute_dataset_loss(const TinyML_DenseLayer *layer, const TinyML_Dataset *dataset) {
    if (dataset->sample_count == 0) {
        return 0.0f;
    }

    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    TinyML_Matrix target = tinyml_matrix_create(1, 1);
    float total_loss = 0.0f;

    for (size_t i = 0; i < dataset->sample_count; ++i) {
        tinyml_matrix_set(&input, 0, 0, tinyml_matrix_get(&dataset->features, i, 0));
        tinyml_matrix_set(&target, 0, 0, tinyml_matrix_get(&dataset->targets, i, 0));

        TinyML_Matrix pred = tinyml_dense_forward(layer, &input);
        total_loss += tinyml_mse_loss(&target, &pred);
        tinyml_matrix_free(&pred);
    }

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);

    return total_loss / (float)dataset->sample_count;
}

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
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);

    if (dataset.sample_count == 0) {
        fprintf(stderr, "Failed to load dataset: %s\n", config.data_path);
        tinyml_dense_free(&layer);
        return 1;
    }

    tinyml_dataset_split(
    &dataset,
    config.validation_split,
    config.shuffle,
    config.split_seed,
    &train_dataset,
    &val_dataset
);

TinyML_NormalizationStats norm_stats = tinyml_fit_normalization(&train_dataset);
tinyml_apply_normalization(&train_dataset, &norm_stats);
tinyml_apply_normalization(&val_dataset, &norm_stats);

    if (train_dataset.sample_count == 0) {
        fprintf(stderr, "Training split is empty.\n");
        tinyml_dataset_free(&dataset);
        tinyml_dense_free(&layer);
        return 1;
    }

    tinyml_matrix_set(&layer.weights, 0, 0, 0.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    float final_train_loss = 0.0f;
    float final_val_loss = 0.0f;
    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    TinyML_Matrix target = tinyml_matrix_create(1, 1);

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < train_dataset.sample_count; ++i) {
            tinyml_matrix_set(&input, 0, 0, tinyml_matrix_get(&train_dataset.features, i, 0));
            tinyml_matrix_set(&target, 0, 0, tinyml_matrix_get(&train_dataset.targets, i, 0));

            epoch_loss += tinyml_train_step_dense(&layer, &input, &target, config.learning_rate);
        }

        epoch_loss /= (float)train_dataset.sample_count;
        final_train_loss = epoch_loss;
        final_val_loss = tinyml_compute_dataset_loss(&layer, &val_dataset);

        if ((epoch + 1) % 20 == 0 || epoch == 0) {
            printf(
                "Epoch %d/%d - Train Loss: %.6f - Val Loss: %.6f\n",
                epoch + 1,
                config.epochs,
                final_train_loss,
                final_val_loss
            );
        }
    }

    TinyML_Matrix test_input = tinyml_matrix_create(1, 1);
    float normalized_test_x = tinyml_normalize_single_value(4.0f, norm_stats.mean[0], norm_stats.std[0]);
    tinyml_matrix_set(&test_input, 0, 0, normalized_test_x);
    TinyML_Matrix prediction = tinyml_dense_forward(&layer, &test_input);

    printf("Config: %s\n", config_path);
    printf("Dataset: %s\n", config.data_path);
    printf("Prediction for x=4.0: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));
    printf("Normalized x=4.0: %.6f\n", normalized_test_x);
    printf("Learned weight: %.6f\n", tinyml_matrix_get(&layer.weights, 0, 0));
    printf("Learned bias: %.6f\n", tinyml_matrix_get(&layer.bias, 0, 0));

    if (!tinyml_write_training_metrics_json(
        config.metrics_path,
        config.epochs,
        config.learning_rate,
        final_train_loss,
        final_val_loss,
        config.validation_split,
        config.shuffle,
        config.split_seed,
        tinyml_matrix_get(&layer.weights, 0, 0),
        tinyml_matrix_get(&layer.bias, 0, 0))) {
        fprintf(stderr, "Warning: failed to write metrics file: %s\n", config.metrics_path);
    }

    if (!tinyml_save_dense_checkpoint(config.checkpoint_path, &layer)) {
        fprintf(stderr, "Warning: failed to write checkpoint: %s\n", config.checkpoint_path);
    }

    if (!tinyml_save_normalization_stats(config.normalization_path, &norm_stats)) {
    fprintf(stderr, "Warning: failed to write normalization stats: %s\n", config.normalization_path);
    }

    tinyml_matrix_free(&prediction);
    tinyml_matrix_free(&test_input);
    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);
    tinyml_dataset_free(&dataset);
    tinyml_dataset_free(&train_dataset);
    tinyml_dataset_free(&val_dataset);
    tinyml_dense_free(&layer);
    tinyml_normalization_stats_free(&norm_stats);

    return 0;
}