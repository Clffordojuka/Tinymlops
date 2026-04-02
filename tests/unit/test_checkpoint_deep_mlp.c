#include <assert.h>
#include "tinyml.h"

int main(void) {
    size_t hidden_sizes[2] = {3, 2};
    TinyML_DeepMLP mlp = tinyml_deep_mlp_create(2, hidden_sizes, 2, 1, TINYML_ACT_TANH);

    /* layer 0 */
    tinyml_matrix_set(&mlp.layers[0].weights, 0, 0, 1.0f);
    tinyml_matrix_set(&mlp.layers[0].weights, 1, 0, 2.0f);
    tinyml_matrix_set(&mlp.layers[0].weights, 0, 1, 3.0f);
    tinyml_matrix_set(&mlp.layers[0].weights, 1, 1, 4.0f);
    tinyml_matrix_set(&mlp.layers[0].weights, 0, 2, 5.0f);
    tinyml_matrix_set(&mlp.layers[0].weights, 1, 2, 6.0f);
    tinyml_matrix_set(&mlp.layers[0].bias, 0, 0, 0.1f);
    tinyml_matrix_set(&mlp.layers[0].bias, 0, 1, 0.2f);
    tinyml_matrix_set(&mlp.layers[0].bias, 0, 2, 0.3f);

    /* layer 1 */
    tinyml_matrix_set(&mlp.layers[1].weights, 0, 0, 0.7f);
    tinyml_matrix_set(&mlp.layers[1].weights, 1, 0, 0.8f);
    tinyml_matrix_set(&mlp.layers[1].weights, 2, 0, 0.9f);
    tinyml_matrix_set(&mlp.layers[1].weights, 0, 1, 1.1f);
    tinyml_matrix_set(&mlp.layers[1].weights, 1, 1, 1.2f);
    tinyml_matrix_set(&mlp.layers[1].weights, 2, 1, 1.3f);
    tinyml_matrix_set(&mlp.layers[1].bias, 0, 0, 0.4f);
    tinyml_matrix_set(&mlp.layers[1].bias, 0, 1, 0.5f);

    /* layer 2 */
    tinyml_matrix_set(&mlp.layers[2].weights, 0, 0, 1.4f);
    tinyml_matrix_set(&mlp.layers[2].weights, 1, 0, 1.5f);
    tinyml_matrix_set(&mlp.layers[2].bias, 0, 0, 0.6f);

    assert(tinyml_save_deep_mlp_checkpoint("test_deep_mlp_checkpoint.txt", &mlp) == 1);

    TinyML_DeepMLP loaded;
    assert(tinyml_load_deep_mlp_checkpoint("test_deep_mlp_checkpoint.txt", &loaded) == 1);

    assert(loaded.num_layers == 3);
    assert(loaded.hidden_activation == TINYML_ACT_TANH);

    assert(loaded.layers[0].input_dim == 2);
    assert(loaded.layers[0].output_dim == 3);
    assert(loaded.layers[1].input_dim == 3);
    assert(loaded.layers[1].output_dim == 2);
    assert(loaded.layers[2].input_dim == 2);
    assert(loaded.layers[2].output_dim == 1);

    assert(tinyml_matrix_get(&loaded.layers[0].weights, 0, 0) == 1.0f);
    assert(tinyml_matrix_get(&loaded.layers[0].weights, 1, 2) == 6.0f);
    assert(tinyml_matrix_get(&loaded.layers[1].weights, 2, 1) == 1.3f);
    assert(tinyml_matrix_get(&loaded.layers[2].weights, 1, 0) == 1.5f);
    assert(tinyml_matrix_get(&loaded.layers[2].bias, 0, 0) == 0.6f);

    tinyml_deep_mlp_free(&mlp);
    tinyml_deep_mlp_free(&loaded);

    return 0;
}