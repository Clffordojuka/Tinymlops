#include <assert.h>
#include "tinyml.h"

#ifndef TINYML_SOURCE_DIR
#define TINYML_SOURCE_DIR "."
#endif

int main(void) {
    const char *csv_path = TINYML_SOURCE_DIR "/data/samples/linear.csv";
    TinyML_Dataset dataset = tinyml_dataset_load_csv(csv_path);

    assert(dataset.sample_count == 4);
    assert(dataset.feature_count == 1);

    assert(tinyml_matrix_get(&dataset.features, 0, 0) == 1.0f);
    assert(tinyml_matrix_get(&dataset.targets, 0, 0) == 2.0f);

    assert(tinyml_matrix_get(&dataset.features, 3, 0) == 4.0f);
    assert(tinyml_matrix_get(&dataset.targets, 3, 0) == 8.0f);

    tinyml_dataset_free(&dataset);
    return 0;
}