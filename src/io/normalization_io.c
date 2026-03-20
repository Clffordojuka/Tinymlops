#include <stdio.h>
#include <stdlib.h>
#include "tinyml.h"

int tinyml_save_normalization_stats(const char *path, const TinyML_NormalizationStats *stats) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL || stats == NULL) {
        return 0;
    }

    fprintf(fp, "feature_count=%zu\n", stats->feature_count);
    for (size_t i = 0; i < stats->feature_count; ++i) {
        fprintf(fp, "mean_%zu=%f\n", i, stats->mean[i]);
        fprintf(fp, "std_%zu=%f\n", i, stats->std[i]);
    }

    fclose(fp);
    return 1;
}

int tinyml_load_normalization_stats(const char *path, TinyML_NormalizationStats *stats) {
    FILE *fp = fopen(path, "r");
    size_t feature_count = 0;

    if (fp == NULL || stats == NULL) {
        return 0;
    }

    if (fscanf(fp, "feature_count=%zu\n", &feature_count) != 1) {
        fclose(fp);
        return 0;
    }

    *stats = tinyml_normalization_stats_create(feature_count);

    for (size_t i = 0; i < feature_count; ++i) {
        size_t idx_mean = 0;
        size_t idx_std = 0;

        if (fscanf(fp, "mean_%zu=%f\n", &idx_mean, &stats->mean[i]) != 2 || idx_mean != i) {
            fclose(fp);
            tinyml_normalization_stats_free(stats);
            return 0;
        }

        if (fscanf(fp, "std_%zu=%f\n", &idx_std, &stats->std[i]) != 2 || idx_std != i) {
            fclose(fp);
            tinyml_normalization_stats_free(stats);
            return 0;
        }
    }

    fclose(fp);
    return 1;
}