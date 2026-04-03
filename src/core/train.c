#include <stdlib.h>
#include "tinyml.h"

float tinyml_train_step_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda,
    TinyML_OptimizerType optimizer,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon)
{
    TinyML_Matrix pred = tinyml_dense_forward(layer, input);
    float loss = tinyml_mse_loss(target, &pred);
    TinyML_Matrix grad = tinyml_mse_loss_gradient(target, &pred);
    TinyML_Matrix grad_input = tinyml_dense_backward(
        layer,
        input,
        &grad,
        learning_rate,
        l2_lambda,
        optimizer,
        adam_beta1,
        adam_beta2,
        adam_epsilon);

    tinyml_matrix_free(&pred);
    tinyml_matrix_free(&grad);
    tinyml_matrix_free(&grad_input);

    return loss;
}

float tinyml_train_batch_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *inputs,
    const TinyML_Matrix *targets,
    float learning_rate,
    float l2_lambda,
    TinyML_OptimizerType optimizer,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon)
{
    TinyML_Matrix pred = tinyml_dense_forward(layer, inputs);
    float loss = tinyml_mse_loss(targets, &pred);
    TinyML_Matrix grad = tinyml_mse_loss_gradient(targets, &pred);
    TinyML_Matrix grad_input = tinyml_dense_backward(
        layer,
        inputs,
        &grad,
        learning_rate,
        l2_lambda,
        optimizer,
        adam_beta1,
        adam_beta2,
        adam_epsilon);

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
    float learning_rate,
    float l2_lambda,
    TinyML_OptimizerType optimizer,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon)
{
    float total_loss = 0.0f;
    TinyML_Matrix input = tinyml_matrix_create(1, inputs->cols);
    TinyML_Matrix target = tinyml_matrix_create(1, targets->cols);

    for (size_t i = 0; i < sample_count; ++i)
    {
        for (size_t j = 0; j < inputs->cols; ++j)
        {
            tinyml_matrix_set(&input, 0, j, tinyml_matrix_get(inputs, i, j));
        }

        for (size_t j = 0; j < targets->cols; ++j)
        {
            tinyml_matrix_set(&target, 0, j, tinyml_matrix_get(targets, i, j));
        }

        total_loss += tinyml_train_step_dense(
            layer,
            &input,
            &target,
            learning_rate,
            l2_lambda,
            optimizer,
            adam_beta1,
            adam_beta2,
            adam_epsilon);
    }

    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);

    return (sample_count > 0) ? total_loss / (float)sample_count : 0.0f;
}

float tinyml_train_step_mlp(
    TinyML_MLP *mlp,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda,
    TinyML_OptimizerType optimizer,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon)
{
    TinyML_Matrix hidden_linear = tinyml_dense_forward(&mlp->hidden, input);
    TinyML_Matrix hidden_activated = tinyml_matrix_copy(&hidden_linear);
    tinyml_matrix_apply_activation(&hidden_activated, mlp->hidden_activation);

    TinyML_Matrix output = tinyml_dense_forward(&mlp->output, &hidden_activated);
    float loss = tinyml_mse_loss(target, &output);

    TinyML_Matrix grad_output = tinyml_mse_loss_gradient(target, &output);
    TinyML_Matrix grad_hidden = tinyml_dense_backward(
        &mlp->output,
        &hidden_activated,
        &grad_output,
        learning_rate,
        l2_lambda,
        optimizer,
        adam_beta1,
        adam_beta2,
        adam_epsilon);

    TinyML_Matrix activation_grad = tinyml_matrix_copy(&hidden_linear);
    tinyml_matrix_apply_activation_derivative_inplace(&activation_grad, mlp->hidden_activation);

    for (size_t r = 0; r < grad_hidden.rows; ++r)
    {
        for (size_t c = 0; c < grad_hidden.cols; ++c)
        {
            float g = tinyml_matrix_get(&grad_hidden, r, c);
            float ad = tinyml_matrix_get(&activation_grad, r, c);
            tinyml_matrix_set(&grad_hidden, r, c, g * ad);
        }
    }

    TinyML_Matrix grad_input = tinyml_dense_backward(
        &mlp->hidden,
        input,
        &grad_hidden,
        learning_rate,
        l2_lambda,
        optimizer,
        adam_beta1,
        adam_beta2,
        adam_epsilon);

    tinyml_matrix_free(&hidden_linear);
    tinyml_matrix_free(&hidden_activated);
    tinyml_matrix_free(&output);
    tinyml_matrix_free(&grad_output);
    tinyml_matrix_free(&grad_hidden);
    tinyml_matrix_free(&activation_grad);
    tinyml_matrix_free(&grad_input);

    return loss;
}

float tinyml_train_step_deep_mlp(
    TinyML_DeepMLP *mlp,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda,
    TinyML_OptimizerType optimizer,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon)
{
    TinyML_Matrix *linear_outputs = (TinyML_Matrix *)malloc(sizeof(TinyML_Matrix) * mlp->num_layers);
    TinyML_Matrix *layer_outputs = (TinyML_Matrix *)malloc(sizeof(TinyML_Matrix) * (mlp->num_layers + 1));
    TinyML_Matrix grad;
    float loss = 0.0f;

    if (linear_outputs == NULL || layer_outputs == NULL)
    {
        free(linear_outputs);
        free(layer_outputs);
        return 0.0f;
    }

    layer_outputs[0] = tinyml_matrix_copy(input);

    for (size_t i = 0; i < mlp->num_layers; ++i)
    {
        linear_outputs[i] = tinyml_dense_forward(&mlp->layers[i], &layer_outputs[i]);
        layer_outputs[i + 1] = tinyml_matrix_copy(&linear_outputs[i]);

        if (i + 1 < mlp->num_layers)
        {
            tinyml_matrix_apply_activation(&layer_outputs[i + 1], mlp->hidden_activation);
        }
    }

    loss = tinyml_mse_loss(target, &layer_outputs[mlp->num_layers]);
    grad = tinyml_mse_loss_gradient(target, &layer_outputs[mlp->num_layers]);

    for (size_t rev = mlp->num_layers; rev > 0; --rev)
    {
        size_t i = rev - 1;
        TinyML_Matrix next_grad = tinyml_dense_backward(
            &mlp->layers[i],
            &layer_outputs[i],
            &grad,
            learning_rate,
            l2_lambda,
            optimizer,
            adam_beta1,
            adam_beta2,
            adam_epsilon);

        tinyml_matrix_free(&grad);

        if (i > 0)
        {
            TinyML_Matrix activation_grad = tinyml_matrix_copy(&linear_outputs[i - 1]);
            tinyml_matrix_apply_activation_derivative_inplace(&activation_grad, mlp->hidden_activation);

            for (size_t r = 0; r < next_grad.rows; ++r)
            {
                for (size_t c = 0; c < next_grad.cols; ++c)
                {
                    float g = tinyml_matrix_get(&next_grad, r, c);
                    float a = tinyml_matrix_get(&activation_grad, r, c);
                    tinyml_matrix_set(&next_grad, r, c, g * a);
                }
            }

            tinyml_matrix_free(&activation_grad);
        }

        grad = next_grad;
    }

    tinyml_matrix_free(&grad);

    for (size_t i = 0; i < mlp->num_layers; ++i)
    {
        tinyml_matrix_free(&linear_outputs[i]);
    }
    for (size_t i = 0; i < mlp->num_layers + 1; ++i)
    {
        tinyml_matrix_free(&layer_outputs[i]);
    }

    free(linear_outputs);
    free(layer_outputs);

    return loss;
}