#include <stdio.h>
#include "tinyml.h"

int tinyml_save_dense_checkpoint(const char *path, const TinyML_DenseLayer *layer) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL || layer == NULL) {
        return 0;
    }

    fprintf(fp, "input_dim=%zu\n", layer->input_dim);
    fprintf(fp, "output_dim=%zu\n", layer->output_dim);

    for (size_t i = 0; i < layer->input_dim; ++i) {
        for (size_t o = 0; o < layer->output_dim; ++o) {
            fprintf(fp, "weight_%zu_%zu=%f\n", i, o, tinyml_matrix_get(&layer->weights, i, o));
        }
    }

    for (size_t o = 0; o < layer->output_dim; ++o) {
        fprintf(fp, "bias_%zu=%f\n", o, tinyml_matrix_get(&layer->bias, 0, o));
    }

    fclose(fp);
    return 1;
}

int tinyml_load_dense_checkpoint(const char *path, TinyML_DenseLayer *layer) {
    FILE *fp = fopen(path, "r");
    size_t input_dim = 0;
    size_t output_dim = 0;

    if (fp == NULL || layer == NULL) {
        return 0;
    }

    if (fscanf(fp, "input_dim=%zu\n", &input_dim) != 1 ||
        fscanf(fp, "output_dim=%zu\n", &output_dim) != 1) {
        fclose(fp);
        return 0;
    }

    *layer = tinyml_dense_create(input_dim, output_dim);

    for (size_t i = 0; i < input_dim; ++i) {
        for (size_t o = 0; o < output_dim; ++o) {
            size_t read_i = 0;
            size_t read_o = 0;
            float weight = 0.0f;

            if (fscanf(fp, "weight_%zu_%zu=%f\n", &read_i, &read_o, &weight) != 3 ||
                read_i != i || read_o != o) {
                fclose(fp);
                tinyml_dense_free(layer);
                return 0;
            }

            tinyml_matrix_set(&layer->weights, i, o, weight);
        }
    }

    for (size_t o = 0; o < output_dim; ++o) {
        size_t read_o = 0;
        float bias = 0.0f;

        if (fscanf(fp, "bias_%zu=%f\n", &read_o, &bias) != 2 || read_o != o) {
            fclose(fp);
            tinyml_dense_free(layer);
            return 0;
        }

        tinyml_matrix_set(&layer->bias, 0, o, bias);
    }

    fclose(fp);
    return 1;
}