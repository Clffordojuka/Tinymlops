#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_MLP mlp = tinyml_mlp_create(2, 2, 1, TINYML_ACT_RELU);

    tinyml_matrix_set(&mlp.hidden.weights, 0, 0, 1.0f);
    tinyml_matrix_set(&mlp.hidden.weights, 1, 0, 1.0f);
    tinyml_matrix_set(&mlp.hidden.weights, 0, 1, 0.5f);
    tinyml_matrix_set(&mlp.hidden.weights, 1, 1, 0.5f);

    tinyml_matrix_set(&mlp.hidden.bias, 0, 0, 0.0f);
    tinyml_matrix_set(&mlp.hidden.bias, 0, 1, 0.0f);

    tinyml_matrix_set(&mlp.output.weights, 0, 0, 1.0f);
    tinyml_matrix_set(&mlp.output.weights, 1, 0, 1.0f);
    tinyml_matrix_set(&mlp.output.bias, 0, 0, 0.0f);

    TinyML_Matrix input = tinyml_matrix_create(1, 2);
    tinyml_matrix_set(&input, 0, 0, 2.0f);
    tinyml_matrix_set(&input, 0, 1, 2.0f);

    TinyML_Matrix out = tinyml_mlp_forward(&mlp, &input);

    assert(tinyml_matrix_get(&out, 0, 0) > 5.9f);
    assert(tinyml_matrix_get(&out, 0, 0) < 6.1f);

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&out);
    tinyml_mlp_free(&mlp);

    return 0;
}