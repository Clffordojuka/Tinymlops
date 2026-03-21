#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer layer = tinyml_dense_create(2, 1);

    tinyml_matrix_set(&layer.weights, 0, 0, 1.0f);
    tinyml_matrix_set(&layer.weights, 0, 1, 3.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 1.0f);

    TinyML_Matrix input = tinyml_matrix_create(1, 2);
    tinyml_matrix_set(&input, 0, 0, 4.0f);
    tinyml_matrix_set(&input, 0, 1, 1.0f);

    TinyML_Matrix pred = tinyml_dense_forward(&layer, &input);

    assert(tinyml_matrix_get(&pred, 0, 0) > 7.9f);
    assert(tinyml_matrix_get(&pred, 0, 0) < 8.1f);

    tinyml_matrix_free(&pred);
    tinyml_matrix_free(&input);
    tinyml_dense_free(&layer);

    return 0;
}