#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "tinyml.h"

static void tinyml_trim_newline(char *s) {
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r')) {
        s[len - 1] = '\0';
        len--;
    }
}

TinyML_TrainConfig tinyml_default_train_config(void) {
    TinyML_TrainConfig config;

    snprintf(config.data_path, sizeof(config.data_path), "data/samples/linear.csv");
    config.epochs = 200;
    config.learning_rate = 0.01f;
    config.batch_size = 2;
    snprintf(config.metrics_path, sizeof(config.metrics_path), "metrics/train_metrics.json");
    snprintf(config.checkpoint_path, sizeof(config.checkpoint_path), "models/checkpoints/linear_model.txt");
    config.validation_split = 0.25f;
    config.test_split = 0.20f;
    config.shuffle = 1;
    config.split_seed = 42u;
    config.patience = 20;
    config.min_delta = 0.000001f;
    config.save_best_only = 1;
    snprintf(config.eval_metrics_path, sizeof(config.eval_metrics_path), "metrics/eval_metrics.json");
    snprintf(config.normalization_path, sizeof(config.normalization_path), "models/checkpoints/normalization_stats.txt");
    snprintf(config.lr_schedule, sizeof(config.lr_schedule), "constant");
    config.lr_step_size = 50;
    config.lr_decay = 0.5f;
    config.l2_lambda = 0.0f;
    snprintf(config.model_type, sizeof(config.model_type), "linear");
    config.hidden_dim = 8;

    return config;
}

int tinyml_load_train_config(const char *path, TinyML_TrainConfig *config) {
    FILE *fp = fopen(path, "r");
    char line[512];

    if (config == NULL) {
        return 0;
    }

    *config = tinyml_default_train_config();

    if (fp == NULL) {
        return 0;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        char key[128];
        char value[256];

        tinyml_trim_newline(line);

        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }

        if (sscanf(line, "%127[^=]=%255[^\n]", key, value) == 2) {
            if (strcmp(key, "data_path") == 0) {
                snprintf(config->data_path, sizeof(config->data_path), "%s", value);
            } else if (strcmp(key, "epochs") == 0) {
                config->epochs = atoi(value);
            } else if (strcmp(key, "learning_rate") == 0) {
                config->learning_rate = (float)atof(value);
            } else if (strcmp(key, "metrics_path") == 0) {
                snprintf(config->metrics_path, sizeof(config->metrics_path), "%s", value);
            } else if (strcmp(key, "checkpoint_path") == 0) {
                snprintf(config->checkpoint_path, sizeof(config->checkpoint_path), "%s", value);
            } else if (strcmp(key, "validation_split") == 0) {
                config->validation_split = (float)atof(value);
            } else if (strcmp(key, "test_split") == 0) {
                config->test_split = (float)atof(value);
            } else if (strcmp(key, "shuffle") == 0) {
                config->shuffle = atoi(value);
            } else if (strcmp(key, "split_seed") == 0) {
                config->split_seed = (unsigned int)strtoul(value, NULL, 10);
            } else if (strcmp(key, "eval_metrics_path") == 0) {
                snprintf(config->eval_metrics_path, sizeof(config->eval_metrics_path), "%s", value);
            } else if (strcmp(key, "normalization_path") == 0) {
                snprintf(config->normalization_path, sizeof(config->normalization_path), "%s", value);
            } else if (strcmp(key, "batch_size") == 0) {
                config->batch_size = (size_t)strtoul(value, NULL, 10);
            } else if (strcmp(key, "patience") == 0) {
                config->patience = atoi(value);
            } else if (strcmp(key, "min_delta") == 0) {
                config->min_delta = (float)atof(value);
            } else if (strcmp(key, "save_best_only") == 0) {
                config->save_best_only = atoi(value);
            } else if (strcmp(key, "lr_schedule") == 0) {
                snprintf(config->lr_schedule, sizeof(config->lr_schedule), "%.31s", value);
            } else if (strcmp(key, "lr_step_size") == 0) {
                config->lr_step_size = atoi(value);
            } else if (strcmp(key, "lr_decay") == 0) {
                config->lr_decay = (float)atof(value);
            } else if (strcmp(key, "l2_lambda") == 0) {
                config->l2_lambda = (float)atof(value);
            } else if (strcmp(key, "model_type") == 0) {
               snprintf(config->model_type, sizeof(config->model_type), "%.31s", value);
            } else if (strcmp(key, "hidden_dim") == 0) {
                config->hidden_dim = (size_t)strtoul(value, NULL, 10);
            }
        }
    }

    fclose(fp);
    return 1;
}