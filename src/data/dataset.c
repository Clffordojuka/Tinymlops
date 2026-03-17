#include "tinyml.h"

TinyML_Dataset tinyml_dataset_create(size_t sample_count, size_t feature_count) {
    TinyML_Dataset dataset;
    dataset.sample_count = sample_count;
    dataset.feature_count = feature_count;
    dataset.features = tinyml_matrix_create(sample_count, feature_count);
    dataset.targets = tinyml_matrix_create(sample_count, 1);
    return dataset;
}

void tinyml_dataset_free(TinyML_Dataset *dataset) {
    if (dataset == NULL) {
        return;
    }

    tinyml_matrix_free(&dataset->features);
    tinyml_matrix_free(&dataset->targets);
    dataset->sample_count = 0;
    dataset->feature_count = 0;
}