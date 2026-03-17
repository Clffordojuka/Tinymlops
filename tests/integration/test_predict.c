#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);

    tinyml_matrix_set(&layer.weights, 0, 0, 2.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    float y = tinyml_predict_dense_single(&layer, 4.0f);

    assert(y > 7.9f);
    assert(y < 8.1f);

    tinyml_dense_free(&layer);
    return 0;
}