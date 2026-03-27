#include <math.h>
#include "tinyml.h"

TinyML_DenseLayer tinyml_dense_create(size_t input_dim, size_t output_dim) {
    TinyML_DenseLayer layer;
    layer.input_dim = input_dim;
    layer.output_dim = output_dim;
    layer.weights = tinyml_matrix_create(input_dim, output_dim);
    layer.bias = tinyml_matrix_create(1, output_dim);
    return layer;
}

void tinyml_dense_free(TinyML_DenseLayer *layer) {
    if (layer == NULL) {
        return;
    }

    tinyml_matrix_free(&layer->weights);
    tinyml_matrix_free(&layer->bias);
    layer->input_dim = 0;
    layer->output_dim = 0;
}

TinyML_Matrix tinyml_dense_forward(const TinyML_DenseLayer *layer, const TinyML_Matrix *input) {
    TinyML_Matrix output = tinyml_matrix_create(input->rows, layer->output_dim);

    for (size_t r = 0; r < input->rows; ++r) {
        for (size_t c = 0; c < layer->output_dim; ++c) {
            float sum = tinyml_matrix_get(&layer->bias, 0, c);

            for (size_t k = 0; k < layer->input_dim; ++k) {
                float x = tinyml_matrix_get(input, r, k);
                float w = tinyml_matrix_get(&layer->weights, k, c);
                sum += x * w;
            }

            tinyml_matrix_set(&output, r, c, sum);
        }
    }

    return output;
}

TinyML_Matrix tinyml_dense_backward(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *input,
    const TinyML_Matrix *grad_output,
    float learning_rate,
    float l2_lambda
) {
    TinyML_Matrix input_t = tinyml_matrix_transpose(input);
    TinyML_Matrix dW = tinyml_matrix_multiply(&input_t, grad_output);
    TinyML_Matrix grad_input = tinyml_matrix_create(input->rows, layer->input_dim);

    for (size_t r = 0; r < input->rows; ++r) {
        for (size_t i = 0; i < layer->input_dim; ++i) {
            float sum = 0.0f;
            for (size_t o = 0; o < layer->output_dim; ++o) {
                float go = tinyml_matrix_get(grad_output, r, o);
                float w = tinyml_matrix_get(&layer->weights, i, o);
                sum += go * w;
            }
            tinyml_matrix_set(&grad_input, r, i, sum);
        }
    }

    for (size_t i = 0; i < layer->input_dim; ++i) {
        for (size_t o = 0; o < layer->output_dim; ++o) {
            float w = tinyml_matrix_get(&layer->weights, i, o);
            float grad = tinyml_matrix_get(&dW, i, o);
            float reg_grad = l2_lambda * w;
            tinyml_matrix_set(&layer->weights, i, o, w - learning_rate * (grad + reg_grad));
        }
    }

    for (size_t o = 0; o < layer->output_dim; ++o) {
        float grad_b = 0.0f;
        for (size_t r = 0; r < grad_output->rows; ++r) {
            grad_b += tinyml_matrix_get(grad_output, r, o);
        }

        float b = tinyml_matrix_get(&layer->bias, 0, o);
        tinyml_matrix_set(&layer->bias, 0, o, b - learning_rate * grad_b);
    }

    tinyml_matrix_free(&input_t);
    tinyml_matrix_free(&dW);

    return grad_input;
}

int tinyml_dense_parameter_count(const TinyML_DenseLayer *layer) {
    if (layer == NULL) {
        return 0;
    }

    return (int)(layer->input_dim * layer->output_dim + layer->output_dim);
}

float tinyml_dense_weight_l2_norm(const TinyML_DenseLayer *layer) {
    float sum_sq = 0.0f;

    if (layer == NULL) {
        return 0.0f;
    }

    for (size_t i = 0; i < layer->input_dim; ++i) {
        for (size_t o = 0; o < layer->output_dim; ++o) {
            float w = tinyml_matrix_get(&layer->weights, i, o);
            sum_sq += w * w;
        }
    }

    return sqrtf(sum_sq);
}

float tinyml_dense_max_abs_weight(const TinyML_DenseLayer *layer) {
    float max_abs = 0.0f;

    if (layer == NULL) {
        return 0.0f;
    }

    for (size_t i = 0; i < layer->input_dim; ++i) {
        for (size_t o = 0; o < layer->output_dim; ++o) {
            float w = tinyml_matrix_get(&layer->weights, i, o);
            float abs_w = (w < 0.0f) ? -w : w;
            if (abs_w > max_abs) {
                max_abs = abs_w;
            }
        }
    }

    return max_abs;
}

float tinyml_dense_bias_l2_norm(const TinyML_DenseLayer *layer) {
    float sum_sq = 0.0f;

    if (layer == NULL) {
        return 0.0f;
    }

    for (size_t o = 0; o < layer->output_dim; ++o) {
        float b = tinyml_matrix_get(&layer->bias, 0, o);
        sum_sq += b * b;
    }

    return sqrtf(sum_sq);
}