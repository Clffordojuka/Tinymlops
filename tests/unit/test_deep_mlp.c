#include <assert.h>
#include "tinyml.h"

int main(void) {
    size_t hidden_sizes[2] = {4, 3};
    TinyML_DeepMLP mlp = tinyml_deep_mlp_create(2, hidden_sizes, 2, 1, TINYML_ACT_RELU);

    assert(mlp.num_layers == 3);
    assert(mlp.layers != NULL);
    assert(mlp.hidden_activation == TINYML_ACT_RELU);

    assert(mlp.layers[0].input_dim == 2);
    assert(mlp.layers[0].output_dim == 4);

    assert(mlp.layers[1].input_dim == 4);
    assert(mlp.layers[1].output_dim == 3);

    assert(mlp.layers[2].input_dim == 3);
    assert(mlp.layers[2].output_dim == 1);

    tinyml_deep_mlp_free(&mlp);
    return 0;
}