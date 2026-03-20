#include <assert.h>
#include "tinyml.h"

static TinyML_Dataset make_dataset(void) {
    TinyML_Dataset dataset = tinyml_dataset_create(8, 1);

    for (size_t i = 0; i < 8; ++i) {
        tinyml_matrix_set(&dataset.features, i, 0, (float)(i + 1));
        tinyml_matrix_set(&dataset.targets, i, 0, (float)(i + 1) * 2.0f);
    }

    return dataset;
}

int main(void) {
    TinyML_Dataset dataset = make_dataset();

    TinyML_Dataset train_a, val_a;
    TinyML_Dataset train_b, val_b;
    TinyML_Dataset train_c, val_c;
    TinyML_Dataset train_d, val_d;

    tinyml_dataset_split(&dataset, 0.25f, 1, 42u, &train_a, &val_a);
    tinyml_dataset_split(&dataset, 0.25f, 1, 42u, &train_b, &val_b);
    tinyml_dataset_split(&dataset, 0.25f, 1, 7u, &train_c, &val_c);
    tinyml_dataset_split(&dataset, 0.25f, 0, 42u, &train_d, &val_d);

    assert(train_a.sample_count == 6);
    assert(val_a.sample_count == 2);

    for (size_t i = 0; i < train_a.sample_count; ++i) {
        assert(tinyml_matrix_get(&train_a.features, i, 0) ==
               tinyml_matrix_get(&train_b.features, i, 0));
    }

    for (size_t i = 0; i < val_a.sample_count; ++i) {
        assert(tinyml_matrix_get(&val_a.features, i, 0) ==
               tinyml_matrix_get(&val_b.features, i, 0));
    }

    {
        int any_diff = 0;
        for (size_t i = 0; i < train_a.sample_count; ++i) {
            if (tinyml_matrix_get(&train_a.features, i, 0) !=
                tinyml_matrix_get(&train_c.features, i, 0)) {
                any_diff = 1;
                break;
            }
        }
        assert(any_diff == 1);
    }

    assert(tinyml_matrix_get(&train_d.features, 0, 0) == 1.0f);
    assert(tinyml_matrix_get(&train_d.features, 1, 0) == 2.0f);
    assert(tinyml_matrix_get(&train_d.features, 2, 0) == 3.0f);
    assert(tinyml_matrix_get(&train_d.features, 3, 0) == 4.0f);
    assert(tinyml_matrix_get(&train_d.features, 4, 0) == 5.0f);
    assert(tinyml_matrix_get(&train_d.features, 5, 0) == 6.0f);

    assert(tinyml_matrix_get(&val_d.features, 0, 0) == 7.0f);
    assert(tinyml_matrix_get(&val_d.features, 1, 0) == 8.0f);

    tinyml_dataset_free(&dataset);
    tinyml_dataset_free(&train_a);
    tinyml_dataset_free(&val_a);
    tinyml_dataset_free(&train_b);
    tinyml_dataset_free(&val_b);
    tinyml_dataset_free(&train_c);
    tinyml_dataset_free(&val_c);
    tinyml_dataset_free(&train_d);
    tinyml_dataset_free(&val_d);

    return 0;
}