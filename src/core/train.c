#include "tinyml.h"

float tinyml_train_step_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate
) {
    TinyML_Matrix pred = tinyml_dense_forward(layer, input);
    float loss = tinyml_mse_loss(target, &pred);

    TinyML_Matrix grad = tinyml_mse_loss_gradient(target, &pred);
    TinyML_Matrix grad_input = tinyml_dense_backward(layer, input, &grad, learning_rate);

    tinyml_matrix_free(&pred);
    tinyml_matrix_free(&grad);
    tinyml_matrix_free(&grad_input);

    return loss;
}

float tinyml_train_epoch_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *inputs,
    const TinyML_Matrix *targets,
    size_t sample_count,
    float learning_rate
) {
    float total_loss = 0.0f;

    for (size_t i = 0; i < sample_count; ++i) {
        total_loss += tinyml_train_step_dense(layer, &inputs[i], &targets[i], learning_rate);
    }

    if (sample_count == 0) {
        return 0.0f;
    }

    return total_loss / (float)sample_count;
}