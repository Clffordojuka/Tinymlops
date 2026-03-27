#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_MLP mlp = tinyml_mlp_create(2, 4, 1);

    assert(mlp.hidden.input_dim == 2);
    assert(mlp.hidden.output_dim == 4);
    assert(mlp.output.input_dim == 4);
    assert(mlp.output.output_dim == 1);

    tinyml_mlp_free(&mlp);
    return 0;
}