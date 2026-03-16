#include "tinyml.h"

float tinyml_mse_loss(const TinyML_Matrix *y_true, const TinyML_Matrix *y_pred) {
    float sum = 0.0f;
    size_t count = y_true->rows * y_true->cols;

    for (size_t r = 0; r < y_true->rows; ++r) {
        for (size_t c = 0; c < y_true->cols; ++c) {
            float diff = tinyml_matrix_get(y_true, r, c) - tinyml_matrix_get(y_pred, r, c);
            sum += diff * diff;
        }
    }

    return (count > 0) ? (sum / (float)count) : 0.0f;
}

TinyML_Matrix tinyml_mse_loss_gradient(const TinyML_Matrix *y_true, const TinyML_Matrix *y_pred) {
    TinyML_Matrix grad = tinyml_matrix_create(y_true->rows, y_true->cols);
    size_t count = y_true->rows * y_true->cols;

    if (count == 0) {
        return grad;
    }

    for (size_t r = 0; r < y_true->rows; ++r) {
        for (size_t c = 0; c < y_true->cols; ++c) {
            float diff = tinyml_matrix_get(y_pred, r, c) - tinyml_matrix_get(y_true, r, c);
            tinyml_matrix_set(&grad, r, c, (2.0f * diff) / (float)count);
        }
    }

    return grad;
}