#include "tinyml.h"

TinyML_Model tinyml_model_create(size_t input_dim, size_t output_dim, TinyML_Activation activation) {
    TinyML_Model model;
    model.dense = tinyml_dense_create(input_dim, output_dim);
    model.activation = activation;
    return model;
}

void tinyml_model_free(TinyML_Model *model) {
    if (model == NULL) {
        return;
    }

    tinyml_dense_free(&model->dense);
    model->activation = TINYML_ACT_NONE;
}

TinyML_Matrix tinyml_model_forward(const TinyML_Model *model, const TinyML_Matrix *input) {
    TinyML_Matrix output = tinyml_dense_forward(&model->dense, input);

    if (model->activation == TINYML_ACT_RELU) {
        tinyml_matrix_apply_relu(&output);
    }

    return output;
}