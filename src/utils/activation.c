#include <string.h>
#include <math.h>
#include "tinyml.h"

float tinyml_relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float tinyml_relu_derivative(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

float tinyml_tanh(float x) {
    return tanhf(x);
}

float tinyml_tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - (t * t);
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

void tinyml_matrix_apply_tanh(TinyML_Matrix *matrix) {
    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            float value = tinyml_matrix_get(matrix, r, c);
            tinyml_matrix_set(matrix, r, c, tinyml_tanh(value));
        }
    }
}

void tinyml_matrix_apply_tanh_derivative_inplace(TinyML_Matrix *matrix) {
    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            float value = tinyml_matrix_get(matrix, r, c);
            tinyml_matrix_set(matrix, r, c, tinyml_tanh_derivative(value));
        }
    }
}

TinyML_Activation tinyml_activation_from_string(const char *name) {
    if (name == NULL) {
        return TINYML_ACT_RELU;
    }

    if (strcmp(name, "relu") == 0) {
        return TINYML_ACT_RELU;
    }
    if (strcmp(name, "tanh") == 0) {
        return TINYML_ACT_TANH;
    }
    if (strcmp(name, "linear") == 0) {
        return TINYML_ACT_LINEAR;
    }

    return TINYML_ACT_RELU;
}

const char *tinyml_activation_to_string(TinyML_Activation activation) {
    switch (activation) {
        case TINYML_ACT_RELU:
            return "relu";
        case TINYML_ACT_TANH:
            return "tanh";
        case TINYML_ACT_LINEAR:
            return "linear";
        case TINYML_ACT_NONE:
        default:
            return "none";
    }
}

void tinyml_matrix_apply_activation(TinyML_Matrix *matrix, TinyML_Activation activation) {
    if (activation == TINYML_ACT_RELU) {
        tinyml_matrix_apply_relu(matrix);
    } else if (activation == TINYML_ACT_TANH) {
        tinyml_matrix_apply_tanh(matrix);
    }
}

void tinyml_matrix_apply_activation_derivative_inplace(TinyML_Matrix *matrix, TinyML_Activation activation) {
    if (activation == TINYML_ACT_RELU) {
        tinyml_matrix_apply_relu_derivative_inplace(matrix);
    } else if (activation == TINYML_ACT_TANH) {
        tinyml_matrix_apply_tanh_derivative_inplace(matrix);
    } else {
        for (size_t r = 0; r < matrix->rows; ++r) {
            for (size_t c = 0; c < matrix->cols; ++c) {
                tinyml_matrix_set(matrix, r, c, 1.0f);
            }
        }
    }
}