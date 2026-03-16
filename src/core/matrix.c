#include <stdlib.h>
#include "tinyml.h"

TinyML_Matrix tinyml_matrix_create(size_t rows, size_t cols) {
    TinyML_Matrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;
    matrix.data = NULL;

    if (rows == 0 || cols == 0) {
        return matrix;
    }

    matrix.data = (float *)calloc(rows * cols, sizeof(float));
    return matrix;
}

void tinyml_matrix_free(TinyML_Matrix *matrix) {
    if (matrix == NULL) {
        return;
    }

    free(matrix->data);
    matrix->data = NULL;
    matrix->rows = 0;
    matrix->cols = 0;
}

float tinyml_matrix_get(const TinyML_Matrix *matrix, size_t row, size_t col) {
    return matrix->data[row * matrix->cols + col];
}

void tinyml_matrix_set(TinyML_Matrix *matrix, size_t row, size_t col, float value) {
    matrix->data[row * matrix->cols + col] = value;
}

void tinyml_matrix_fill(TinyML_Matrix *matrix, float value) {
    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            tinyml_matrix_set(matrix, r, c, value);
        }
    }
}

TinyML_Matrix tinyml_matrix_copy(const TinyML_Matrix *matrix) {
    TinyML_Matrix copy = tinyml_matrix_create(matrix->rows, matrix->cols);

    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            tinyml_matrix_set(&copy, r, c, tinyml_matrix_get(matrix, r, c));
        }
    }

    return copy;
}

TinyML_Matrix tinyml_matrix_subtract(const TinyML_Matrix *a, const TinyML_Matrix *b) {
    TinyML_Matrix out = tinyml_matrix_create(a->rows, a->cols);

    for (size_t r = 0; r < a->rows; ++r) {
        for (size_t c = 0; c < a->cols; ++c) {
            float value = tinyml_matrix_get(a, r, c) - tinyml_matrix_get(b, r, c);
            tinyml_matrix_set(&out, r, c, value);
        }
    }

    return out;
}

TinyML_Matrix tinyml_matrix_transpose(const TinyML_Matrix *matrix) {
    TinyML_Matrix out = tinyml_matrix_create(matrix->cols, matrix->rows);

    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            tinyml_matrix_set(&out, c, r, tinyml_matrix_get(matrix, r, c));
        }
    }

    return out;
}

TinyML_Matrix tinyml_matrix_multiply(const TinyML_Matrix *a, const TinyML_Matrix *b) {
    TinyML_Matrix out = tinyml_matrix_create(a->rows, b->cols);

    for (size_t r = 0; r < a->rows; ++r) {
        for (size_t c = 0; c < b->cols; ++c) {
            float sum = 0.0f;
            for (size_t k = 0; k < a->cols; ++k) {
                sum += tinyml_matrix_get(a, r, k) * tinyml_matrix_get(b, k, c);
            }
            tinyml_matrix_set(&out, r, c, sum);
        }
    }

    return out;
}

void tinyml_matrix_scale_inplace(TinyML_Matrix *matrix, float scalar) {
    for (size_t r = 0; r < matrix->rows; ++r) {
        for (size_t c = 0; c < matrix->cols; ++c) {
            float value = tinyml_matrix_get(matrix, r, c) * scalar;
            tinyml_matrix_set(matrix, r, c, value);
        }
    }
}