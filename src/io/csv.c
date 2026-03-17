#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinyml.h"

TinyML_Dataset tinyml_dataset_load_csv(const char *path) {
    TinyML_Dataset dataset = tinyml_dataset_create(0, 0);
    FILE *fp = fopen(path, "r");

    if (fp == NULL) {
        return dataset;
    }

    char line[256];
    size_t rows = 0;

    /* skip header */
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return dataset;
    }

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

    dataset = tinyml_dataset_create(rows, 1);

    size_t row = 0;
    while (fgets(line, sizeof(line), fp) != NULL && row < rows) {
        float x = 0.0f;
        float y = 0.0f;

        if (sscanf(line, "%f,%f", &x, &y) == 2) {
            tinyml_matrix_set(&dataset.features, row, 0, x);
            tinyml_matrix_set(&dataset.targets, row, 0, y);
            row++;
        }
    }

    fclose(fp);
    return dataset;
}