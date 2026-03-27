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

static void tinyml_copy_dataset_rows(
    const TinyML_Dataset *src,
    const size_t *indices,
    size_t start,
    size_t count,
    TinyML_Dataset *dst
) {
    for (size_t i = 0; i < count; ++i) {
        size_t src_idx = indices[start + i];

        for (size_t j = 0; j < src->feature_count; ++j) {
            tinyml_matrix_set(
                &dst->features,
                i,
                j,
                tinyml_matrix_get(&src->features, src_idx, j)
            );
        }

        tinyml_matrix_set(
            &dst->targets,
            i,
            0,
            tinyml_matrix_get(&src->targets, src_idx, 0)
        );
    }
}

void tinyml_dataset_split_three_way(
    const TinyML_Dataset *dataset,
    float validation_split,
    float test_split,
    int shuffle,
    unsigned int split_seed,
    TinyML_Dataset *train_dataset,
    TinyML_Dataset *val_dataset,
    TinyML_Dataset *test_dataset
) {
    if (dataset == NULL || train_dataset == NULL || val_dataset == NULL || test_dataset == NULL) {
        return;
    }

    if (validation_split < 0.0f) {
        validation_split = 0.0f;
    }
    if (test_split < 0.0f) {
        test_split = 0.0f;
    }
    if (validation_split + test_split > 0.9f) {
        float scale = 0.9f / (validation_split + test_split);
        validation_split *= scale;
        test_split *= scale;
    }

    size_t total = dataset->sample_count;
    size_t val_count = (size_t)(total * validation_split);
    size_t test_count = (size_t)(total * test_split);
    size_t train_count = total - val_count - test_count;

    *train_dataset = tinyml_dataset_create(train_count, dataset->feature_count);
    *val_dataset = tinyml_dataset_create(val_count, dataset->feature_count);
    *test_dataset = tinyml_dataset_create(test_count, dataset->feature_count);

    size_t *indices = (size_t *)malloc(total * sizeof(size_t));
    if (indices == NULL) {
        return;
    }

    for (size_t i = 0; i < total; ++i) {
        indices[i] = i;
    }

    if (shuffle) {
        srand(split_seed);
        for (size_t i = total; i > 1; --i) {
            size_t j = (size_t)(rand() % i);
            tinyml_swap_size_t(&indices[i - 1], &indices[j]);
        }
    }

    tinyml_copy_dataset_rows(dataset, indices, 0, train_count, train_dataset);
    tinyml_copy_dataset_rows(dataset, indices, train_count, val_count, val_dataset);
    tinyml_copy_dataset_rows(dataset, indices, train_count + val_count, test_count, test_dataset);

    free(indices);
}

void tinyml_dataset_split(
    const TinyML_Dataset *dataset,
    float validation_split,
    int shuffle,
    unsigned int split_seed,
    TinyML_Dataset *train_dataset,
    TinyML_Dataset *val_dataset
) {
    TinyML_Dataset test_dataset;
    tinyml_dataset_split_three_way(
        dataset,
        validation_split,
        0.0f,
        shuffle,
        split_seed,
        train_dataset,
        val_dataset,
        &test_dataset
    );
    tinyml_dataset_free(&test_dataset);
}