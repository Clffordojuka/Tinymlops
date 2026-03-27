#include <stdio.h>
#include <string.h>
#include "tinyml.h"

static int tinyml_write_dense_block(FILE *fp, const char *prefix, const TinyML_DenseLayer *layer) {
    if (fp == NULL || layer == NULL) {
        return 0;
    }

    fprintf(fp, "%s_input_dim=%zu\n", prefix, layer->input_dim);
    fprintf(fp, "%s_output_dim=%zu\n", prefix, layer->output_dim);

    for (size_t i = 0; i < layer->input_dim; ++i) {
        for (size_t o = 0; o < layer->output_dim; ++o) {
            fprintf(
                fp,
                "%s_weight_%zu_%zu=%f\n",
                prefix,
                i,
                o,
                tinyml_matrix_get(&layer->weights, i, o)
            );
        }
    }

    for (size_t o = 0; o < layer->output_dim; ++o) {
        fprintf(fp, "%s_bias_%zu=%f\n", prefix, o, tinyml_matrix_get(&layer->bias, 0, o));
    }

    return 1;
}

static int tinyml_read_dense_block(FILE *fp, const char *prefix, TinyML_DenseLayer *layer) {
    size_t input_dim = 0;
    size_t output_dim = 0;
    char expected_key[128];
    char read_key[128];
    float value = 0.0f;

    if (fp == NULL || layer == NULL) {
        return 0;
    }

    snprintf(expected_key, sizeof(expected_key), "%s_input_dim", prefix);
    if (fscanf(fp, "%127[^=]=%zu\n", read_key, &input_dim) != 2 || strcmp(read_key, expected_key) != 0) {
        return 0;
    }

    snprintf(expected_key, sizeof(expected_key), "%s_output_dim", prefix);
    if (fscanf(fp, "%127[^=]=%zu\n", read_key, &output_dim) != 2 || strcmp(read_key, expected_key) != 0) {
        return 0;
    }

    *layer = tinyml_dense_create(input_dim, output_dim);

    for (size_t i = 0; i < input_dim; ++i) {
        for (size_t o = 0; o < output_dim; ++o) {
            snprintf(expected_key, sizeof(expected_key), "%s_weight_%zu_%zu", prefix, i, o);
            if (fscanf(fp, "%127[^=]=%f\n", read_key, &value) != 2 || strcmp(read_key, expected_key) != 0) {
                tinyml_dense_free(layer);
                return 0;
            }
            tinyml_matrix_set(&layer->weights, i, o, value);
        }
    }

    for (size_t o = 0; o < output_dim; ++o) {
        snprintf(expected_key, sizeof(expected_key), "%s_bias_%zu", prefix, o);
        if (fscanf(fp, "%127[^=]=%f\n", read_key, &value) != 2 || strcmp(read_key, expected_key) != 0) {
            tinyml_dense_free(layer);
            return 0;
        }
        tinyml_matrix_set(&layer->bias, 0, o, value);
    }

    return 1;
}

int tinyml_save_dense_checkpoint(const char *path, const TinyML_DenseLayer *layer) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL || layer == NULL) {
        return 0;
    }

    fprintf(fp, "model_type=dense\n");
    if (!tinyml_write_dense_block(fp, "dense", layer)) {
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

int tinyml_load_dense_checkpoint(const char *path, TinyML_DenseLayer *layer) {
    FILE *fp = fopen(path, "r");
    char key[128];
    char model_type[128];

    if (fp == NULL || layer == NULL) {
        return 0;
    }

    if (fscanf(fp, "%127[^=]=%127s\n", key, model_type) != 2 ||
        strcmp(key, "model_type") != 0 ||
        strcmp(model_type, "dense") != 0) {
        fclose(fp);
        return 0;
    }

    if (!tinyml_read_dense_block(fp, "dense", layer)) {
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

int tinyml_save_mlp_checkpoint(const char *path, const TinyML_MLP *mlp) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL || mlp == NULL) {
        return 0;
    }

    fprintf(fp, "model_type=mlp\n");

    if (!tinyml_write_dense_block(fp, "hidden", &mlp->hidden)) {
        fclose(fp);
        return 0;
    }

    if (!tinyml_write_dense_block(fp, "output", &mlp->output)) {
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 1;
}

int tinyml_load_mlp_checkpoint(const char *path, TinyML_MLP *mlp) {
    FILE *fp = fopen(path, "r");
    char key[128];
    char model_type[128];

    if (fp == NULL || mlp == NULL) {
        return 0;
    }

    if (fscanf(fp, "%127[^=]=%127s\n", key, model_type) != 2 ||
        strcmp(key, "model_type") != 0 ||
        strcmp(model_type, "mlp") != 0) {
        fclose(fp);
        return 0;
    }

    if (!tinyml_read_dense_block(fp, "hidden", &mlp->hidden)) {
        fclose(fp);
        return 0;
    }

    if (!tinyml_read_dense_block(fp, "output", &mlp->output)) {
        fclose(fp);
        tinyml_dense_free(&mlp->hidden);
        return 0;
    }

    fclose(fp);
    return 1;
}