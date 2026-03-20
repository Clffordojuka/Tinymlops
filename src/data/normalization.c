#include <stdlib.h>
#include <math.h>
#include "tinyml.h"

TinyML_NormalizationStats tinyml_normalization_stats_create(size_t feature_count) {
    TinyML_NormalizationStats stats;
    stats.feature_count = feature_count;
    stats.mean = (float *)calloc(feature_count, sizeof(float));
    stats.std = (float *)calloc(feature_count, sizeof(float));
    return stats;
}

void tinyml_normalization_stats_free(TinyML_NormalizationStats *stats) {
    if (stats == NULL) {
        return;
    }

    free(stats->mean);
    free(stats->std);
    stats->mean = NULL;
    stats->std = NULL;
    stats->feature_count = 0;
}

TinyML_NormalizationStats tinyml_fit_normalization(const TinyML_Dataset *dataset) {
    TinyML_NormalizationStats stats = tinyml_normalization_stats_create(dataset->feature_count);

    if (dataset->sample_count == 0 || dataset->feature_count == 0) {
        return stats;
    }

    for (size_t j = 0; j < dataset->feature_count; ++j) {
        float sum = 0.0f;
        float var_sum = 0.0f;

        for (size_t i = 0; i < dataset->sample_count; ++i) {
            sum += tinyml_matrix_get(&dataset->features, i, j);
        }

        stats.mean[j] = sum / (float)dataset->sample_count;

        for (size_t i = 0; i < dataset->sample_count; ++i) {
            float value = tinyml_matrix_get(&dataset->features, i, j);
            float diff = value - stats.mean[j];
            var_sum += diff * diff;
        }

        stats.std[j] = sqrtf(var_sum / (float)dataset->sample_count);
        if (stats.std[j] < 1e-8f) {
            stats.std[j] = 1.0f;
        }
    }

    return stats;
}

void tinyml_apply_normalization(TinyML_Dataset *dataset, const TinyML_NormalizationStats *stats) {
    if (dataset == NULL || stats == NULL) {
        return;
    }

    for (size_t i = 0; i < dataset->sample_count; ++i) {
        for (size_t j = 0; j < dataset->feature_count; ++j) {
            float value = tinyml_matrix_get(&dataset->features, i, j);
            float normalized = (value - stats->mean[j]) / stats->std[j];
            tinyml_matrix_set(&dataset->features, i, j, normalized);
        }
    }
}

float tinyml_normalize_single_value(float value, float mean, float std) {
    if (std < 1e-8f) {
        std = 1.0f;
    }
    return (value - mean) / std;
}