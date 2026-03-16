#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_Matrix y_true = tinyml_matrix_create(1, 2);
    TinyML_Matrix y_pred = tinyml_matrix_create(1, 2);

    tinyml_matrix_set(&y_true, 0, 0, 1.0f);
    tinyml_matrix_set(&y_true, 0, 1, 3.0f);

    tinyml_matrix_set(&y_pred, 0, 0, 2.0f);
    tinyml_matrix_set(&y_pred, 0, 1, 1.0f);

    float loss = tinyml_mse_loss(&y_true, &y_pred);

    assert(loss == 2.5f);

    tinyml_matrix_free(&y_true);
    tinyml_matrix_free(&y_pred);

    return 0;
}