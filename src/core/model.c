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
    } else if (model->activation == TINYML_ACT_TANH) {
        tinyml_matrix_apply_tanh(&output);
    }

    return output;
}

TinyML_MLP tinyml_mlp_create(
    size_t input_dim,
    size_t hidden_dim,
    size_t output_dim,
    TinyML_Activation hidden_activation
) {
    TinyML_MLP mlp;
    mlp.hidden = tinyml_dense_create(input_dim, hidden_dim);
    mlp.output = tinyml_dense_create(hidden_dim, output_dim);
    mlp.hidden_activation = hidden_activation;
    return mlp;
}

void tinyml_mlp_free(TinyML_MLP *mlp) {
    if (mlp == NULL) {
        return;
    }

    tinyml_dense_free(&mlp->hidden);
    tinyml_dense_free(&mlp->output);
    mlp->hidden_activation = TINYML_ACT_NONE;
}

TinyML_Matrix tinyml_mlp_forward(const TinyML_MLP *mlp, const TinyML_Matrix *input) {
    TinyML_Matrix hidden_out = tinyml_dense_forward(&mlp->hidden, input);
    tinyml_matrix_apply_activation(&hidden_out, mlp->hidden_activation);

    TinyML_Matrix output = tinyml_dense_forward(&mlp->output, &hidden_out);
    tinyml_matrix_free(&hidden_out);

    return output;
}