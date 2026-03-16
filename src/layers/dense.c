#include <stdlib.h>
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