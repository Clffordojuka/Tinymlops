#include <assert.h>
#include "tinyml.h"

int main(void) {
    size_t hidden_sizes[2] = {2, 2};
    TinyML_DeepMLP mlp = tinyml_deep_mlp_create(2, hidden_sizes, 2, 1, TINYML_ACT_RELU);

    /* layer 0: 2 -> 2 */
    tinyml_matrix_set(&mlp.layers[0].weights, 0, 0, 1.0f);
    tinyml_matrix_set(&mlp.layers[0].weights, 1, 0, 1.0f);
    tinyml_matrix_set(&mlp.layers[0].weights, 0, 1, 0.5f);
    tinyml_matrix_set(&mlp.layers[0].weights, 1, 1, 0.5f);
    tinyml_matrix_set(&mlp.layers[0].bias, 0, 0, 0.0f);
    tinyml_matrix_set(&mlp.layers[0].bias, 0, 1, 0.0f);

    /* layer 1: 2 -> 2 */
    tinyml_matrix_set(&mlp.layers[1].weights, 0, 0, 1.0f);
    tinyml_matrix_set(&mlp.layers[1].weights, 1, 0, 0.0f);
    tinyml_matrix_set(&mlp.layers[1].weights, 0, 1, 0.0f);
    tinyml_matrix_set(&mlp.layers[1].weights, 1, 1, 1.0f);
    tinyml_matrix_set(&mlp.layers[1].bias, 0, 0, 0.0f);
    tinyml_matrix_set(&mlp.layers[1].bias, 0, 1, 0.0f);

    /* output: 2 -> 1 */
    tinyml_matrix_set(&mlp.layers[2].weights, 0, 0, 1.0f);
    tinyml_matrix_set(&mlp.layers[2].weights, 1, 0, 1.0f);
    tinyml_matrix_set(&mlp.layers[2].bias, 0, 0, 0.0f);

    TinyML_Matrix input = tinyml_matrix_create(1, 2);
    tinyml_matrix_set(&input, 0, 0, 2.0f);
    tinyml_matrix_set(&input, 0, 1, 2.0f);

    TinyML_Matrix out = tinyml_deep_mlp_forward(&mlp, &input);

    assert(tinyml_matrix_get(&out, 0, 0) > 5.9f);
    assert(tinyml_matrix_get(&out, 0, 0) < 6.1f);

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&out);
    tinyml_deep_mlp_free(&mlp);

    return 0;
}