#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinyml.h"

static size_t tinyml_count_csv_columns(const char *line) {
    size_t count = 1;

    for (const char *p = line; *p != '\0'; ++p) {
        if (*p == ',') {
            count++;
        }
    }

    return count;
}

TinyML_Dataset tinyml_dataset_load_csv(const char *path) {
    TinyML_Dataset dataset = tinyml_dataset_create(0, 0);
    FILE *fp = fopen(path, "r");

    if (fp == NULL) {
        return dataset;
    }

    char line[1024];
    size_t rows = 0;
    size_t total_columns = 0;
    size_t feature_count = 0;

    /* read header */
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return dataset;
    }

    total_columns = tinyml_count_csv_columns(line);
    if (total_columns < 2) {
        fclose(fp);
        return dataset;
    }

    feature_count = total_columns - 1;

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strlen(line) > 1) {
            rows++;
        }
    }

    rewind(fp);

    /* skip header again */
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return dataset;
    }

    dataset = tinyml_dataset_create(rows, feature_count);

    size_t row = 0;
    while (fgets(line, sizeof(line), fp) != NULL && row < rows) {
        char *token = strtok(line, ",");
        size_t col = 0;

        while (token != NULL && col < total_columns) {
            float value = (float)atof(token);

            if (col < feature_count) {
                tinyml_matrix_set(&dataset.features, row, col, value);
            } else {
                tinyml_matrix_set(&dataset.targets, row, 0, value);
            }

            token = strtok(NULL, ",");
            col++;
        }

        if (col == total_columns) {
            row++;
        }
    }

    fclose(fp);
    return dataset;
}