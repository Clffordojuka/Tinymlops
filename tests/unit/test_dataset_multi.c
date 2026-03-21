#include <assert.h>
#include "tinyml.h"

#ifndef TINYML_SOURCE_DIR
#define TINYML_SOURCE_DIR "."
#endif

int main(void) {
    TinyML_Dataset dataset = tinyml_dataset_load_csv(TINYML_SOURCE_DIR "/data/samples/linear_multi.csv");

    assert(dataset.sample_count == 6);
    assert(dataset.feature_count == 2);

    assert(tinyml_matrix_get(&dataset.features, 0, 0) == 1.0f);
    assert(tinyml_matrix_get(&dataset.features, 0, 1) == 2.0f);
    assert(tinyml_matrix_get(&dataset.targets, 0, 0) == 8.0f);

    assert(tinyml_matrix_get(&dataset.features, 5, 0) == 6.0f);
    assert(tinyml_matrix_get(&dataset.features, 5, 1) == 3.0f);
    assert(tinyml_matrix_get(&dataset.targets, 5, 0) == 21.0f);

    tinyml_dataset_free(&dataset);
    return 0;
}