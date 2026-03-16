#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_Matrix matrix = tinyml_matrix_create(1, 3);

    tinyml_matrix_set(&matrix, 0, 0, -2.0f);
    tinyml_matrix_set(&matrix, 0, 1, 0.0f);
    tinyml_matrix_set(&matrix, 0, 2, 5.0f);

    tinyml_matrix_apply_relu(&matrix);

    assert(tinyml_matrix_get(&matrix, 0, 0) == 0.0f);
    assert(tinyml_matrix_get(&matrix, 0, 1) == 0.0f);
    assert(tinyml_matrix_get(&matrix, 0, 2) == 5.0f);

    tinyml_matrix_free(&matrix);
    return 0;
}