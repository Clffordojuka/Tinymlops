#include <assert.h>
#include "tinyml.h"

int main(void)
{
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);
    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    TinyML_Matrix target = tinyml_matrix_create(1, 1);

    tinyml_matrix_set(&layer.weights, 0, 0, 0.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    tinyml_matrix_set(&input, 0, 0, 1.0f);
    tinyml_matrix_set(&target, 0, 0, 2.0f);

    TinyML_Matrix pred_before = tinyml_dense_forward(&layer, &input);
    float loss_before = tinyml_mse_loss(&target, &pred_before);

    TinyML_Matrix grad = tinyml_mse_loss_gradient(&target, &pred_before);
    TinyML_Matrix grad_input = tinyml_dense_backward(
        &layer,
        &input,
        &grad,
        0.1f,
        0.0f,
        TINYML_OPT_SGD,
        0.9f,
        0.999f,
        0.000001f);
    TinyML_Matrix pred_after = tinyml_dense_forward(&layer, &input);
    float loss_after = tinyml_mse_loss(&target, &pred_after);

    assert(loss_after < loss_before);

    tinyml_matrix_free(&pred_before);
    tinyml_matrix_free(&grad);
    tinyml_matrix_free(&grad_input);
    tinyml_matrix_free(&pred_after);
    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);
    tinyml_dense_free(&layer);

    return 0;
}