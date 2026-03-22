#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer layer = tinyml_dense_create(2, 1);

    tinyml_matrix_set(&layer.weights, 0, 0, 1.5f);
    tinyml_matrix_set(&layer.weights, 1, 0, -0.5f);
    tinyml_matrix_set(&layer.bias, 0, 0, 2.0f);

    assert(tinyml_save_dense_checkpoint("test_multi_checkpoint.txt", &layer) == 1);

    TinyML_DenseLayer loaded;
    assert(tinyml_load_dense_checkpoint("test_multi_checkpoint.txt", &loaded) == 1);

    assert(loaded.input_dim == 2);
    assert(loaded.output_dim == 1);

    assert(tinyml_matrix_get(&loaded.weights, 0, 0) == 1.5f);
    assert(tinyml_matrix_get(&loaded.weights, 1, 0) == -0.5f);
    assert(tinyml_matrix_get(&loaded.bias, 0, 0) == 2.0f);

    tinyml_dense_free(&layer);
    tinyml_dense_free(&loaded);

    return 0;
}