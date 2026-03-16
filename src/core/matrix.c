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