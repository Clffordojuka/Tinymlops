#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_MLP mlp = tinyml_mlp_create(2, 3, 1, TINYML_ACT_RELU);

    tinyml_matrix_set(&mlp.hidden.weights, 0, 0, 1.0f);
    tinyml_matrix_set(&mlp.hidden.weights, 1, 0, 2.0f);
    tinyml_matrix_set(&mlp.hidden.weights, 0, 1, 3.0f);
    tinyml_matrix_set(&mlp.hidden.weights, 1, 1, 4.0f);
    tinyml_matrix_set(&mlp.hidden.weights, 0, 2, 5.0f);
    tinyml_matrix_set(&mlp.hidden.weights, 1, 2, 6.0f);

    tinyml_matrix_set(&mlp.hidden.bias, 0, 0, 0.1f);
    tinyml_matrix_set(&mlp.hidden.bias, 0, 1, 0.2f);
    tinyml_matrix_set(&mlp.hidden.bias, 0, 2, 0.3f);

    tinyml_matrix_set(&mlp.output.weights, 0, 0, 0.7f);
    tinyml_matrix_set(&mlp.output.weights, 1, 0, 0.8f);
    tinyml_matrix_set(&mlp.output.weights, 2, 0, 0.9f);
    tinyml_matrix_set(&mlp.output.bias, 0, 0, 1.5f);

    assert(tinyml_save_mlp_checkpoint("test_mlp_checkpoint.txt", &mlp) == 1);

    TinyML_MLP loaded;
    assert(tinyml_load_mlp_checkpoint("test_mlp_checkpoint.txt", &loaded) == 1);

    assert(loaded.hidden.input_dim == 2);
    assert(loaded.hidden.output_dim == 3);
    assert(loaded.output.input_dim == 3);
    assert(loaded.output.output_dim == 1);
    assert(loaded.hidden_activation == TINYML_ACT_RELU);

    assert(tinyml_matrix_get(&loaded.hidden.weights, 0, 0) == 1.0f);
    assert(tinyml_matrix_get(&loaded.hidden.weights, 1, 2) == 6.0f);
    assert(tinyml_matrix_get(&loaded.hidden.bias, 0, 1) == 0.2f);
    assert(tinyml_matrix_get(&loaded.output.weights, 2, 0) == 0.9f);
    assert(tinyml_matrix_get(&loaded.output.bias, 0, 0) == 1.5f);

    tinyml_mlp_free(&mlp);
    tinyml_mlp_free(&loaded);

    return 0;
}