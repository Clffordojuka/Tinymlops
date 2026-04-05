#include <assert.h>
#include "tinyml.h"

int main(void) {
    TinyML_Dataset dataset = tinyml_dataset_create(5, 1);

    for (size_t i = 0; i < 5; ++i) {
        tinyml_matrix_set(&dataset.features, i, 0, (float)(i + 1));
        tinyml_matrix_set(&dataset.targets, i, 0, (float)(2 * (i + 1)));
    }

    TinyML_Dataset train_dataset;
    TinyML_Dataset val_dataset;
    TinyML_Dataset test_dataset;

    tinyml_dataset_split_three_way(
        &dataset,
        0.2f,
        0.2f,
        1,
        42,
        &train_dataset,
        &val_dataset,
        &test_dataset
    );

    assert(train_dataset.sample_count + val_dataset.sample_count + test_dataset.sample_count == 5);
    assert(train_dataset.sample_count > 0);
    assert(val_dataset.sample_count > 0);
    assert(test_dataset.sample_count > 0);

    tinyml_dataset_free(&dataset);
    tinyml_dataset_free(&train_dataset);
    tinyml_dataset_free(&val_dataset);
    tinyml_dataset_free(&test_dataset);

    return 0;
}