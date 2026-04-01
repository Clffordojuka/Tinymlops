#include <stdio.h>
#include <stdlib.h>
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
            fprintf(fp, "%s_weight_%zu_%zu=%f\n",
                    prefix, i, o, tinyml_matrix_get(&layer->weights, i, o));
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
    fprintf(fp, "hidden_activation=%d\n", (int)mlp->hidden_activation);

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
    int hidden_activation = 0;

    if (fp == NULL || mlp == NULL) {
        return 0;
    }

    if (fscanf(fp, "%127[^=]=%127s\n", key, model_type) != 2 ||
        strcmp(key, "model_type") != 0 ||
        strcmp(model_type, "mlp") != 0) {
        fclose(fp);
        return 0;
    }

    if (fscanf(fp, "%127[^=]=%d\n", key, &hidden_activation) != 2 ||
        strcmp(key, "hidden_activation") != 0) {
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

    mlp->hidden_activation = (TinyML_Activation)hidden_activation;

    fclose(fp);
    return 1;
}

int tinyml_save_deep_mlp_checkpoint(const char *path, const TinyML_DeepMLP *mlp) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL || mlp == NULL || mlp->layers == NULL) {
        return 0;
    }

    fprintf(fp, "model_type=deep_mlp\n");
    fprintf(fp, "hidden_activation=%d\n", (int)mlp->hidden_activation);
    fprintf(fp, "num_layers=%zu\n", mlp->num_layers);

    for (size_t i = 0; i < mlp->num_layers; ++i) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "layer%zu", i);

        if (!tinyml_write_dense_block(fp, prefix, &mlp->layers[i])) {
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}

int tinyml_load_deep_mlp_checkpoint(const char *path, TinyML_DeepMLP *mlp) {
    FILE *fp = fopen(path, "r");
    char key[128];
    char model_type[128];
    int hidden_activation = 0;
    size_t num_layers = 0;

    if (fp == NULL || mlp == NULL) {
        return 0;
    }

    mlp->layers = NULL;
    mlp->num_layers = 0;
    mlp->hidden_activation = TINYML_ACT_NONE;

    if (fscanf(fp, "%127[^=]=%127s\n", key, model_type) != 2 ||
        strcmp(key, "model_type") != 0 ||
        strcmp(model_type, "deep_mlp") != 0) {
        fclose(fp);
        return 0;
    }

    if (fscanf(fp, "%127[^=]=%d\n", key, &hidden_activation) != 2 ||
        strcmp(key, "hidden_activation") != 0) {
        fclose(fp);
        return 0;
    }

    if (fscanf(fp, "%127[^=]=%zu\n", key, &num_layers) != 2 ||
        strcmp(key, "num_layers") != 0 ||
        num_layers == 0) {
        fclose(fp);
        return 0;
    }

    mlp->layers = (TinyML_DenseLayer *)malloc(sizeof(TinyML_DenseLayer) * num_layers);
    if (mlp->layers == NULL) {
        fclose(fp);
        return 0;
    }

    mlp->num_layers = num_layers;
    mlp->hidden_activation = (TinyML_Activation)hidden_activation;

    for (size_t i = 0; i < num_layers; ++i) {
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "layer%zu", i);

        if (!tinyml_read_dense_block(fp, prefix, &mlp->layers[i])) {
            for (size_t j = 0; j < i; ++j) {
                tinyml_dense_free(&mlp->layers[j]);
            }
            free(mlp->layers);
            mlp->layers = NULL;
            mlp->num_layers = 0;
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}