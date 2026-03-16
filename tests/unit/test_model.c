#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_Model model = tinyml_model_create(2, 2, TINYML_ACT_RELU);
    TinyML_Matrix input = tinyml_matrix_create(1, 2);

    tinyml_matrix_set(&model.dense.weights, 0, 0, 1.0f);
    tinyml_matrix_set(&model.dense.weights, 1, 0, -3.0f);
    tinyml_matrix_set(&model.dense.weights, 0, 1, 2.0f);
    tinyml_matrix_set(&model.dense.weights, 1, 1, 1.0f);

    tinyml_matrix_set(&model.dense.bias, 0, 0, 0.0f);
    tinyml_matrix_set(&model.dense.bias, 0, 1, -1.0f);

    tinyml_matrix_set(&input, 0, 0, 1.0f);
    tinyml_matrix_set(&input, 0, 1, 2.0f);

    TinyML_Matrix output = tinyml_model_forward(&model, &input);

    assert(output.rows == 1);
    assert(output.cols == 2);

    assert(tinyml_matrix_get(&output, 0, 0) == 0.0f);
    assert(tinyml_matrix_get(&output, 0, 1) == 3.0f);

    tinyml_matrix_free(&output);
    tinyml_matrix_free(&input);
    tinyml_model_free(&model);

    return 0;
}