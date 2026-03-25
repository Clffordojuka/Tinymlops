#include <stdio.h>
#include "tinyml.h"

static float tinyml_compute_dataset_loss(const TinyML_DenseLayer *layer, const TinyML_Dataset *dataset) {
    if (dataset->sample_count == 0) {
        return 0.0f;
    }

    TinyML_Matrix input = tinyml_matrix_create(1, dataset->feature_count);
    TinyML_Matrix target = tinyml_matrix_create(1, 1);
    float total_loss = 0.0f;

    for (size_t i = 0; i < dataset->sample_count; ++i) {
        for (size_t j = 0; j < dataset->feature_count; ++j) {
            tinyml_matrix_set(&input, 0, j, tinyml_matrix_get(&dataset->features, i, j));
        }

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

    if (config.batch_size == 0) {
        config.batch_size = 1;
    }

    TinyML_Dataset dataset = tinyml_dataset_load_csv(config.data_path);
    TinyML_Dataset train_dataset;
    TinyML_Dataset val_dataset;

    if (dataset.sample_count == 0) {
        fprintf(stderr, "Failed to load dataset: %s\n", config.data_path);
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

    if (train_dataset.sample_count == 0) {
        fprintf(stderr, "Training split is empty.\n");
        tinyml_dataset_free(&dataset);
        tinyml_dataset_free(&train_dataset);
        tinyml_dataset_free(&val_dataset);
        return 1;
    }

    TinyML_NormalizationStats norm_stats = tinyml_fit_normalization(&train_dataset);
    tinyml_apply_normalization(&train_dataset, &norm_stats);
    tinyml_apply_normalization(&val_dataset, &norm_stats);

    TinyML_DenseLayer layer = tinyml_dense_create(train_dataset.feature_count, 1);
    tinyml_matrix_set(&layer.weights, 0, 0, 0.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    float final_train_loss = 0.0f;
    float final_val_loss = 0.0f;

    for (int epoch = 0; epoch < config.epochs; ++epoch) {
        float epoch_loss = 0.0f;
        size_t batch_count = 0;

        for (size_t start = 0; start < train_dataset.sample_count; start += config.batch_size) {
            size_t batch_rows = config.batch_size;
            if (start + batch_rows > train_dataset.sample_count) {
                batch_rows = train_dataset.sample_count - start;
            }

            TinyML_Matrix batch_inputs = tinyml_matrix_create(batch_rows, train_dataset.feature_count);
            TinyML_Matrix batch_targets = tinyml_matrix_create(batch_rows, 1);

            for (size_t r = 0; r < batch_rows; ++r) {
                size_t src_row = start + r;

                for (size_t j = 0; j < train_dataset.feature_count; ++j) {
                    tinyml_matrix_set(
                        &batch_inputs,
                        r,
                        j,
                        tinyml_matrix_get(&train_dataset.features, src_row, j)
                    );
                }

                tinyml_matrix_set(
                    &batch_targets,
                    r,
                    0,
                    tinyml_matrix_get(&train_dataset.targets, src_row, 0)
                );
            }

            epoch_loss += tinyml_train_batch_dense(
                &layer,
                &batch_inputs,
                &batch_targets,
                config.learning_rate
            );
            batch_count++;

            tinyml_matrix_free(&batch_inputs);
            tinyml_matrix_free(&batch_targets);
        }

        final_train_loss = (batch_count > 0) ? epoch_loss / (float)batch_count : 0.0f;
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

    printf("Config: %s\n", config_path);
    printf("Dataset: %s\n", config.data_path);

    if (train_dataset.feature_count == 1) {
        TinyML_Matrix test_input = tinyml_matrix_create(1, 1);
        float normalized_test_x = tinyml_normalize_single_value(4.0f, norm_stats.mean[0], norm_stats.std[0]);
        tinyml_matrix_set(&test_input, 0, 0, normalized_test_x);

        TinyML_Matrix prediction = tinyml_dense_forward(&layer, &test_input);

        printf("Prediction for x=4.0: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));
        printf("Normalized x=4.0: %.6f\n", normalized_test_x);

        tinyml_matrix_free(&prediction);
        tinyml_matrix_free(&test_input);
    } else {
        printf("Sample prediction skipped: dataset has %zu features.\n", train_dataset.feature_count);
    }

    for (size_t i = 0; i < layer.input_dim; ++i) {
        for (size_t o = 0; o < layer.output_dim; ++o) {
            printf("Learned weight[%zu,%zu]: %.6f\n",
                   i, o, tinyml_matrix_get(&layer.weights, i, o));
        }
    }

    for (size_t o = 0; o < layer.output_dim; ++o) {
        printf("Learned bias[%zu]: %.6f\n", o, tinyml_matrix_get(&layer.bias, 0, o));
    }

    if (!tinyml_write_training_metrics_json(
            config.metrics_path,
            config.epochs,
            config.learning_rate,
            config.batch_size,
            final_train_loss,
            final_val_loss,
            config.validation_split,
            config.shuffle,
            config.split_seed,
            tinyml_dense_parameter_count(&layer),
            tinyml_dense_weight_l2_norm(&layer),
            tinyml_dense_max_abs_weight(&layer),
            tinyml_dense_bias_l2_norm(&layer))) {
        fprintf(stderr, "Warning: failed to write metrics file: %s\n", config.metrics_path);
    }

    if (!tinyml_save_dense_checkpoint(config.checkpoint_path, &layer)) {
        fprintf(stderr, "Warning: failed to write checkpoint: %s\n", config.checkpoint_path);
    }

    if (!tinyml_save_normalization_stats(config.normalization_path, &norm_stats)) {
        fprintf(stderr, "Warning: failed to write normalization stats: %s\n", config.normalization_path);
    }

    tinyml_normalization_stats_free(&norm_stats);
    tinyml_dataset_free(&dataset);
    tinyml_dataset_free(&train_dataset);
    tinyml_dataset_free(&val_dataset);
    tinyml_dense_free(&layer);

    return 0;
}