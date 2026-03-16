#include "tinyml.h"

float tinyml_relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float tinyml_relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

void tinyml_matrix_apply_relu(TinyML_Matrix *matrix) {
    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            float value = tinyml_matrix_get(matrix, r, c);
            tinyml_matrix_set(matrix, r, c, tinyml_relu(value));
        }
    }
}

void tinyml_matrix_apply_relu_derivative_inplace(TinyML_Matrix *matrix) {
    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            float value = tinyml_matrix_get(matrix, r, c);
            tinyml_matrix_set(matrix, r, c, tinyml_relu_derivative(value));
        }
    }
}