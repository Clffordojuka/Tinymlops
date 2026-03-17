#include <stdio.h>
#include "tinyml.h"

int tinyml_write_training_metrics_json(
    const char *path,
    int epochs,
    float learning_rate,
    float final_loss,
    float weight,
    float bias
) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"epochs\": %d,\n", epochs);
    fprintf(fp, "  \"learning_rate\": %.6f,\n", learning_rate);
    fprintf(fp, "  \"final_loss\": %.6f,\n", final_loss);
    fprintf(fp, "  \"weight\": %.6f,\n", weight);
    fprintf(fp, "  \"bias\": %.6f\n", bias);
    fprintf(fp, "}\n");

    fclose(fp);
    return 1;
}