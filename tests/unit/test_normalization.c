#include <assert.h>
#include <math.h>
#include "tinyml.h"

int main(void) {
    TinyML_Dataset dataset = tinyml_dataset_create(3, 1);

    tinyml_matrix_set(&dataset.features, 0, 0, 1.0f);
    tinyml_matrix_set(&dataset.features, 1, 0, 2.0f);
    tinyml_matrix_set(&dataset.features, 2, 0, 3.0f);

    TinyML_NormalizationStats stats = tinyml_fit_normalization(&dataset);
    tinyml_apply_normalization(&dataset, &stats);

    assert(fabsf(stats.mean[0] - 2.0f) < 1e-5f);
    assert(stats.std[0] > 0.0f);

    {
        float a = tinyml_matrix_get(&dataset.features, 0, 0);
        float b = tinyml_matrix_get(&dataset.features, 1, 0);
        float c = tinyml_matrix_get(&dataset.features, 2, 0);

        assert(a < b);
        assert(b < c);
    }

    assert(tinyml_save_normalization_stats("test_norm.txt", &stats) == 1);

    TinyML_NormalizationStats loaded;
    assert(tinyml_load_normalization_stats("test_norm.txt", &loaded) == 1);
    assert(fabsf(loaded.mean[0] - stats.mean[0]) < 1e-5f);
    assert(fabsf(loaded.std[0] - stats.std[0]) < 1e-5f);

    tinyml_normalization_stats_free(&stats);
    tinyml_normalization_stats_free(&loaded);
    tinyml_dataset_free(&dataset);

    return 0;
}