#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);

    TinyML_Matrix inputs = tinyml_matrix_create(2, 1);
    TinyML_Matrix targets = tinyml_matrix_create(2, 1);

    tinyml_matrix_set(&inputs, 0, 0, 1.0f);
    tinyml_matrix_set(&inputs, 1, 0, 2.0f);

    tinyml_matrix_set(&targets, 0, 0, 2.0f);
    tinyml_matrix_set(&targets, 1, 0, 4.0f);

    tinyml_matrix_set(&layer.weights, 0, 0, 0.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    float before = tinyml_train_batch_dense(&layer, &inputs, &targets, 0.01f, 0.0f);
    float after = tinyml_train_batch_dense(&layer, &inputs, &targets, 0.01f, 0.0f);

    assert(before >= 0.0f);
    assert(after >= 0.0f);

    tinyml_matrix_free(&inputs);
    tinyml_matrix_free(&targets);
    tinyml_dense_free(&layer);

    return 0;
}