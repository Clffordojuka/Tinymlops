#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer saved = tinyml_dense_create(1, 1);
    tinyml_matrix_set(&saved.weights, 0, 0, 1.5f);
    tinyml_matrix_set(&saved.bias, 0, 0, 0.25f);

    int ok_save = tinyml_save_dense_checkpoint("test_model.txt", &saved);
    assert(ok_save == 1);

    TinyML_DenseLayer loaded;
    int ok_load = tinyml_load_dense_checkpoint("test_model.txt", &loaded);
    assert(ok_load == 1);

    assert(loaded.input_dim == 1);
    assert(loaded.output_dim == 1);
    assert(tinyml_matrix_get(&loaded.weights, 0, 0) == 1.5f);
    assert(tinyml_matrix_get(&loaded.bias, 0, 0) == 0.25f);

    tinyml_dense_free(&saved);
    tinyml_dense_free(&loaded);

    return 0;
}