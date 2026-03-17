#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);

    TinyML_Matrix inputs[3];
    TinyML_Matrix targets[3];

    for (int i = 0; i < 3; ++i) {
        inputs[i] = tinyml_matrix_create(1, 1);
        targets[i] = tinyml_matrix_create(1, 1);
    }

    tinyml_matrix_set(&inputs[0], 0, 0, 1.0f);
    tinyml_matrix_set(&targets[0], 0, 0, 2.0f);

    tinyml_matrix_set(&inputs[1], 0, 0, 2.0f);
    tinyml_matrix_set(&targets[1], 0, 0, 4.0f);

    tinyml_matrix_set(&inputs[2], 0, 0, 3.0f);
    tinyml_matrix_set(&targets[2], 0, 0, 6.0f);

    tinyml_matrix_set(&layer.weights, 0, 0, 0.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    TinyML_Matrix test_input = tinyml_matrix_create(1, 1);
    tinyml_matrix_set(&test_input, 0, 0, 4.0f);

    TinyML_Matrix pred_before = tinyml_dense_forward(&layer, &test_input);
    float before_value = tinyml_matrix_get(&pred_before, 0, 0);

    const int epochs = 200;
    const float learning_rate = 0.01f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        tinyml_train_epoch_dense(&layer, inputs, targets, 3, learning_rate);
    }

    TinyML_Matrix pred_after = tinyml_dense_forward(&layer, &test_input);
    float after_value = tinyml_matrix_get(&pred_after, 0, 0);

    assert(before_value < 1.0f);
    assert(after_value > 7.5f);
    assert(after_value < 8.5f);

    tinyml_matrix_free(&pred_before);
    tinyml_matrix_free(&pred_after);
    tinyml_matrix_free(&test_input);

    for (int i = 0; i < 3; ++i) {
        tinyml_matrix_free(&inputs[i]);
        tinyml_matrix_free(&targets[i]);
    }

    tinyml_dense_free(&layer);

    return 0;
}