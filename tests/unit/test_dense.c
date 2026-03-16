#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer layer = tinyml_dense_create(2, 2);
    TinyML_Matrix input = tinyml_matrix_create(1, 2);

    tinyml_matrix_set(&layer.weights, 0, 0, 1.0f);
    tinyml_matrix_set(&layer.weights, 1, 0, 2.0f);
    tinyml_matrix_set(&layer.weights, 0, 1, 3.0f);
    tinyml_matrix_set(&layer.weights, 1, 1, 4.0f);

    tinyml_matrix_set(&layer.bias, 0, 0, 0.5f);
    tinyml_matrix_set(&layer.bias, 0, 1, -0.5f);

    tinyml_matrix_set(&input, 0, 0, 2.0f);
    tinyml_matrix_set(&input, 0, 1, 1.0f);

    TinyML_Matrix output = tinyml_dense_forward(&layer, &input);

    assert(output.rows == 1);
    assert(output.cols == 2);

    assert(tinyml_matrix_get(&output, 0, 0) == 4.5f);
    assert(tinyml_matrix_get(&output, 0, 1) == 9.5f);

    tinyml_matrix_free(&output);
    tinyml_matrix_free(&input);
    tinyml_dense_free(&layer);

    return 0;
}