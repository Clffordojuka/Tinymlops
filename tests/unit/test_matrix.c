#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_Matrix matrix = tinyml_matrix_create(2, 3);

    assert(matrix.rows == 2);
    assert(matrix.cols == 3);
    assert(matrix.data != NULL);

    tinyml_matrix_set(&matrix, 0, 0, 1.5f);
    tinyml_matrix_set(&matrix, 1, 2, 4.0f);

    assert(tinyml_matrix_get(&matrix, 0, 0) == 1.5f);
    assert(tinyml_matrix_get(&matrix, 1, 2) == 4.0f);

    tinyml_matrix_free(&matrix);

    assert(matrix.data == NULL);
    assert(matrix.rows == 0);
    assert(matrix.cols == 0);

    return 0;
}