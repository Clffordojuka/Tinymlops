#include <stdio.h>
#include <string.h>
#include "tinyml.h"

static float tinyml_current_learning_rate(const TinyML_TrainConfig *config, int epoch_index)
{
    if (strcmp(config->lr_schedule, "step") == 0 && config->lr_step_size > 0)
    {
        int num_decays = epoch_index / config->lr_step_size;
        float lr = config->learning_rate;
        for (int i = 0; i < num_decays; ++i)
        {
            lr *= config->lr_decay;
        }
        return lr;
    }
    return config->learning_rate;
}

int main(int argc, char **argv)
{
    const char *config_path = "configs/base/train_linear.cfg";
    TinyML_TrainConfig config = tinyml_default_train_config();

    if (argc > 1)
    {
        config_path = argv[1];
    }

    if (!tinyml_load_train_config(config_path, &config))
    {
        fprintf(stderr, "Failed to load config: %s\n", config_path);
        return 1;
    }

    if (config.batch_size == 0)
    {
        config.batch_size = 1;
    }

    TinyML_Dataset dataset = tinyml_dataset_load_csv(config.data_path);
    TinyML_Dataset train_dataset;
    TinyML_Dataset val_dataset;
    TinyML_Dataset test_dataset;

    if (dataset.sample_count == 0)
    {
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
        &test_dataset);

    if (train_dataset.sample_count == 0)
    {
        fprintf(stderr, "Training split is empty.\n");
        tinyml_dataset_free(&dataset);
        tinyml_dataset_free(&train_dataset);
        tinyml_dataset_free(&val_dataset);
        tinyml_dataset_free(&test_dataset);
        return 1;
    }

    TinyML_NormalizationStats norm_stats = tinyml_fit_normalization(&train_dataset);
    tinyml_apply_normalization(&train_dataset, &norm_stats);
    tinyml_apply_normalization(&val_dataset, &norm_stats);

    TinyML_RuntimeModel model;
    if (!tinyml_runtime_model_init(&model, &config, train_dataset.feature_count))
    {
        fprintf(stderr, "Failed to initialize model.\n");
        return 1;
    }

    float final_train_loss = 0.0f;
    float final_val_loss = 0.0f;
    float best_val_loss = 1e30f;
    float final_learning_rate = config.learning_rate;
    int best_epoch = 0;
    int stopped_early = 0;
    int epochs_without_improvement = 0;

    for (int epoch = 0; epoch < config.epochs; ++epoch)
    {
        float epoch_loss = 0.0f;
        size_t sample_updates = 0;
        float current_learning_rate = tinyml_current_learning_rate(&config, epoch);
        final_learning_rate = current_learning_rate;

        for (size_t start = 0; start < train_dataset.sample_count; start += config.batch_size)
        {
            size_t batch_rows = config.batch_size;
            if (start + batch_rows > train_dataset.sample_count)
            {
                batch_rows = train_dataset.sample_count - start;
            }

            for (size_t r = 0; r < batch_rows; ++r)
            {
                size_t src_row = start + r;
                TinyML_Matrix input = tinyml_matrix_create(1, train_dataset.feature_count);
                TinyML_Matrix target = tinyml_matrix_create(1, 1);

                for (size_t j = 0; j < train_dataset.feature_count; ++j)
                {
                    tinyml_matrix_set(&input, 0, j, tinyml_matrix_get(&train_dataset.features, src_row, j));
                }
                tinyml_matrix_set(&target, 0, 0, tinyml_matrix_get(&train_dataset.targets, src_row, 0));

                epoch_loss += tinyml_runtime_model_train_step(
                    &model,
                    &input,
                    &target,
                    current_learning_rate,
                    config.l2_lambda);
                sample_updates++;

                tinyml_matrix_free(&input);
                tinyml_matrix_free(&target);
            }
        }

        final_train_loss = sample_updates > 0 ? epoch_loss / (float)sample_updates : 0.0f;
        final_val_loss = tinyml_runtime_model_evaluate(&model, &val_dataset);

        if ((best_val_loss - final_val_loss) > config.min_delta)
        {
            best_val_loss = final_val_loss;
            best_epoch = epoch + 1;
            epochs_without_improvement = 0;

            if (config.save_best_only)
            {
                tinyml_runtime_model_save_checkpoint(&model, config.checkpoint_path);
                tinyml_save_normalization_stats(config.normalization_path, &norm_stats);
            }
        }
        else
        {
            epochs_without_improvement++;
        }

        if ((epoch + 1) % 20 == 0 || epoch == 0)
        {
            printf("Epoch %d/%d - LR: %.6f - Train Loss: %.6f - Val Loss: %.6f\n",
                   epoch + 1, config.epochs, current_learning_rate, final_train_loss, final_val_loss);
        }

        if (config.patience > 0 && epochs_without_improvement >= config.patience)
        {
            stopped_early = 1;
            printf("Early stopping triggered at epoch %d\n", epoch + 1);
            break;
        }
    }

    if (config.save_best_only)
    {
        if (!tinyml_runtime_model_save_checkpoint(&model, config.checkpoint_path))
        {
            fprintf(stderr, "Warning: failed to write checkpoint: %s\n", config.checkpoint_path);
        }
        if (!tinyml_save_normalization_stats(config.normalization_path, &norm_stats))
        {
            fprintf(stderr, "Warning: failed to write normalization stats: %s\n", config.normalization_path);
        }
    }
    printf("Config: %s\n", config_path);
    printf("Dataset: %s\n", config.data_path);
    printf("Model type: %s\n", config.model_type);
    printf("Hidden dim: %zu\n", strcmp(config.model_type, "mlp") == 0 ? config.hidden_dim : 0);
    printf("Hidden layers: %s\n", strcmp(config.model_type, "deep_mlp") == 0 ? config.hidden_layers : (strcmp(config.model_type, "mlp") == 0 ? config.hidden_layers : ""));
    printf("Hidden activation: %s\n", (strcmp(config.model_type, "mlp") == 0 || strcmp(config.model_type, "deep_mlp") == 0) ? config.hidden_activation : "none");
    printf("Best epoch: %d\n", best_epoch);
    printf("Best val loss: %.6f\n", best_val_loss);
    printf("Stopped early: %d\n", stopped_early);

    if (train_dataset.feature_count == 1)
    {
        float normalized_test_x = tinyml_normalize_single_value(4.0f, norm_stats.mean[0], norm_stats.std[0]);
        float prediction = tinyml_runtime_model_predict_single(&model, normalized_test_x);
        printf("Prediction for x=4.0: %.6f\n", prediction);
        printf("Normalized x=4.0: %.6f\n", normalized_test_x);
    }
    else
    {
        printf("Sample prediction skipped: dataset has %zu features.\n", train_dataset.feature_count);
    }
    const char *metrics_hidden_layers = "";
    const char *metrics_hidden_activation = "none";
    size_t metrics_hidden_dim = 0;
    size_t metrics_num_layers = 1;

    if (strcmp(config.model_type, "mlp") == 0)
    {
        metrics_hidden_dim = config.hidden_dim;
        metrics_hidden_layers = config.hidden_dim > 0 ? config.hidden_layers : "";
        metrics_hidden_activation = config.hidden_activation;
        metrics_num_layers = 2;
    }
    else if (strcmp(config.model_type, "deep_mlp") == 0)
    {
        size_t parsed_hidden_sizes[16];
        size_t hidden_count = tinyml_parse_hidden_layers(config.hidden_layers, parsed_hidden_sizes, 16);

        metrics_hidden_dim = hidden_count > 0 ? parsed_hidden_sizes[0] : 0;
        metrics_hidden_layers = config.hidden_layers;
        metrics_hidden_activation = config.hidden_activation;
        metrics_num_layers = hidden_count + 1;
    }

    if (!tinyml_write_training_metrics_json(
            config.metrics_path,
            config.epochs,
            config.learning_rate,
            final_learning_rate,
            config.lr_schedule,
            config.lr_step_size,
            config.lr_decay,
            config.l2_lambda,
            config.model_type,
            metrics_hidden_dim,
            metrics_hidden_layers,
            metrics_hidden_activation,
            metrics_num_layers,
            config.batch_size,
            final_train_loss,
            final_val_loss,
            config.validation_split,
            config.shuffle,
            config.split_seed,
            tinyml_runtime_model_parameter_count(&model),
            tinyml_runtime_model_weight_l2_norm(&model),
            tinyml_runtime_model_max_abs_weight(&model),
            tinyml_runtime_model_bias_l2_norm(&model),
            best_val_loss,
            best_epoch,
            stopped_early,
            config.patience,
            config.min_delta,
            config.save_best_only))
    {
        fprintf(stderr, "Warning: failed to write metrics file: %s\n", config.metrics_path);
    }

    tinyml_runtime_model_free(&model);
    tinyml_normalization_stats_free(&norm_stats);
    tinyml_dataset_free(&dataset);
    tinyml_dataset_free(&train_dataset);
    tinyml_dataset_free(&val_dataset);
    tinyml_dataset_free(&test_dataset);

    return 0;
}