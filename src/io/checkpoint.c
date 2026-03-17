#include <stdio.h>
#include "tinyml.h"

int tinyml_save_dense_checkpoint(const char *path, const TinyML_DenseLayer *layer) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp, "input_dim=%zu\n", layer->input_dim);
    fprintf(fp, "output_dim=%zu\n", layer->output_dim);
    fprintf(fp, "weight=%f\n", tinyml_matrix_get(&layer->weights, 0, 0));
    fprintf(fp, "bias=%f\n", tinyml_matrix_get(&layer->bias, 0, 0));

    fclose(fp);
    return 1;
}

int tinyml_load_dense_checkpoint(const char *path, TinyML_DenseLayer *layer) {
    FILE *fp = fopen(path, "r");
    size_t input_dim = 0;
    size_t output_dim = 0;
    float weight = 0.0f;
    float bias = 0.0f;

    if (fp == NULL || layer == NULL) {
        return 0;
    }

    if (fscanf(fp, "input_dim=%zu\n", &input_dim) != 1 ||
        fscanf(fp, "output_dim=%zu\n", &output_dim) != 1 ||
        fscanf(fp, "weight=%f\n", &weight) != 1 ||
        fscanf(fp, "bias=%f\n", &bias) != 1) {
        fclose(fp);
        return 0;
    }

    fclose(fp);

    *layer = tinyml_dense_create(input_dim, output_dim);
    tinyml_matrix_set(&layer->weights, 0, 0, weight);
    tinyml_matrix_set(&layer->bias, 0, 0, bias);

    return 1;
}