#include <assert.h>
#include "tinyml.h"

int main(void) {
    size_t hidden_sizes[2] = {4, 3};
    TinyML_DeepMLP mlp = tinyml_deep_mlp_create(1, hidden_sizes, 2, 1, TINYML_ACT_RELU);

    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    TinyML_Matrix target = tinyml_matrix_create(1, 1);

    tinyml_matrix_set(&input, 0, 0, 2.0f);
    tinyml_matrix_set(&target, 0, 0, 4.0f);

    float before = tinyml_train_step_deep_mlp(
        &mlp,
        &input,
        &target,
        0.01f,
        0.0f,
        TINYML_OPT_SGD,
        0.9f,
        0.999f,
        0.000001f
    );

    float after = tinyml_train_step_deep_mlp(
        &mlp,
        &input,
        &target,
        0.01f,
        0.0f,
        TINYML_OPT_SGD,
        0.9f,
        0.999f,
        0.000001f
    );

    assert(before >= 0.0f);
    assert(after >= 0.0f);

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);
    tinyml_deep_mlp_free(&mlp);

    return 0;
}