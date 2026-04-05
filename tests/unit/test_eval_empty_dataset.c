#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);
    TinyML_Dataset empty_dataset = tinyml_dataset_create(0, 1);

    float loss = tinyml_evaluate_dense(&layer, &empty_dataset);
    assert(loss < 0.0f);

    tinyml_dense_free(&layer);
    tinyml_dataset_free(&empty_dataset);

    return 0;
}