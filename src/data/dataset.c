#include <stdlib.h>
#include "tinyml.h"

static void tinyml_swap_size_t(size_t *a, size_t *b) {
    size_t tmp = *a;
    *a = *b;
    *b = tmp;
}

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

void tinyml_dataset_split(
    const TinyML_Dataset *dataset,
    float validation_split,
    int shuffle,
    unsigned int split_seed,
    TinyML_Dataset *train_dataset,
    TinyML_Dataset *val_dataset
) {
    size_t val_count;
    size_t train_count;
    size_t *indices;

    if (dataset == NULL || train_dataset == NULL || val_dataset == NULL) {
        return;
    }

    if (validation_split < 0.0f) {
        validation_split = 0.0f;
    }
    if (validation_split > 0.9f) {
        validation_split = 0.9f;
    }

    val_count = (size_t)(dataset->sample_count * validation_split);
    train_count = dataset->sample_count - val_count;

    *train_dataset = tinyml_dataset_create(train_count, dataset->feature_count);
    *val_dataset = tinyml_dataset_create(val_count, dataset->feature_count);

    indices = (size_t *)malloc(dataset->sample_count * sizeof(size_t));
    if (indices == NULL) {
        return;
    }

    for (size_t i = 0; i < dataset->sample_count; ++i) {
        indices[i] = i;
    }

    if (shuffle) {
        srand(split_seed);
        for (size_t i = dataset->sample_count; i > 1; --i) {
            size_t j = (size_t)(rand() % i);
            tinyml_swap_size_t(&indices[i - 1], &indices[j]);
        }
    }

    for (size_t i = 0; i < train_count; ++i) {
        size_t src_idx = indices[i];

        for (size_t j = 0; j < dataset->feature_count; ++j) {
            tinyml_matrix_set(
                &train_dataset->features,
                i,
                j,
                tinyml_matrix_get(&dataset->features, src_idx, j)
            );
        }

        tinyml_matrix_set(
            &train_dataset->targets,
            i,
            0,
            tinyml_matrix_get(&dataset->targets, src_idx, 0)
        );
    }

    for (size_t i = 0; i < val_count; ++i) {
        size_t src_idx = indices[train_count + i];

        for (size_t j = 0; j < dataset->feature_count; ++j) {
            tinyml_matrix_set(
                &val_dataset->features,
                i,
                j,
                tinyml_matrix_get(&dataset->features, src_idx, j)
            );
        }

        tinyml_matrix_set(
            &val_dataset->targets,
            i,
            0,
            tinyml_matrix_get(&dataset->targets, src_idx, 0)
        );
    }

    free(indices);
}