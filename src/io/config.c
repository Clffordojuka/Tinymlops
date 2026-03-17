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
    snprintf(config.metrics_path, sizeof(config.metrics_path), "metrics/train_metrics.json");
    snprintf(config.checkpoint_path, sizeof(config.checkpoint_path), "models/checkpoints/linear_model.txt");

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
            }
        }
    }

    fclose(fp);
    return 1;
}