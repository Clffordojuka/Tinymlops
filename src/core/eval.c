#include "tinyml.h"

float tinyml_evaluate_dense(
    const TinyML_DenseLayer *layer,
    const TinyML_Dataset *dataset
) {
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