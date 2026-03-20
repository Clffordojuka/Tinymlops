#include <stdio.h>
#include "tinyml.h"

int tinyml_write_training_metrics_json(
    const char *path,
    int epochs,
    float learning_rate,
    float train_loss,
    float val_loss,
    float validation_split,
    int shuffle,
    unsigned int split_seed,
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
    fprintf(fp, "  \"train_loss\": %.6f,\n", train_loss);
    fprintf(fp, "  \"val_loss\": %.6f,\n", val_loss);
    fprintf(fp, "  \"validation_split\": %.6f,\n", validation_split);
    fprintf(fp, "  \"shuffle\": %d,\n", shuffle);
    fprintf(fp, "  \"split_seed\": %u,\n", split_seed);
    fprintf(fp, "  \"weight\": %.6f,\n", weight);
    fprintf(fp, "  \"bias\": %.6f\n", bias);
    fprintf(fp, "}\n");

    fclose(fp);
    return 1;
}

int tinyml_write_eval_metrics_json(
    const char *path,
    float eval_loss,
    float prediction_x4,
    float weight,
    float bias
) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"eval_loss\": %.6f,\n", eval_loss);
    fprintf(fp, "  \"prediction_x4\": %.6f,\n", prediction_x4);
    fprintf(fp, "  \"weight\": %.6f,\n", weight);
    fprintf(fp, "  \"bias\": %.6f\n", bias);
    fprintf(fp, "}\n");

    fclose(fp);
    return 1;
}