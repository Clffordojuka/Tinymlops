#include "tinyml.h"

float tinyml_evaluate_dense(
    const TinyML_DenseLayer *layer,
    const TinyML_Dataset *dataset
) {
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

float tinyml_predict_dense_single(const TinyML_DenseLayer *layer, float x) {
    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    tinyml_matrix_set(&input, 0, 0, x);

    TinyML_Matrix pred = tinyml_dense_forward(layer, &input);
    float value = tinyml_matrix_get(&pred, 0, 0);

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&pred);

    return value;
}

float tinyml_evaluate_mlp(
    const TinyML_MLP *mlp,
    const TinyML_Dataset *dataset
) {
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

        TinyML_Matrix pred = tinyml_mlp_forward(mlp, &input);
        total_loss += tinyml_mse_loss(&target, &pred);
        tinyml_matrix_free(&pred);
    }

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);

    return total_loss / (float)dataset->sample_count;
}

float tinyml_predict_mlp_single(const TinyML_MLP *mlp, float x) {
    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    tinyml_matrix_set(&input, 0, 0, x);

    TinyML_Matrix pred = tinyml_mlp_forward(mlp, &input);
    float value = tinyml_matrix_get(&pred, 0, 0);

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&pred);

    return value;
}

float tinyml_evaluate_deep_mlp(
    const TinyML_DeepMLP *mlp,
    const TinyML_Dataset *dataset
) {
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

        TinyML_Matrix pred = tinyml_deep_mlp_forward(mlp, &input);
        total_loss += tinyml_mse_loss(&target, &pred);
        tinyml_matrix_free(&pred);
    }

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);

    return total_loss / (float)dataset->sample_count;
}

float tinyml_predict_deep_mlp_single(const TinyML_DeepMLP *mlp, float x) {
    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    tinyml_matrix_set(&input, 0, 0, x);

    TinyML_Matrix pred = tinyml_deep_mlp_forward(mlp, &input);
    float value = tinyml_matrix_get(&pred, 0, 0);

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&pred);

    return value;
}