#include <stdio.h>
#include "tinyml.h"

int tinyml_write_training_metrics_json(
    const char *path,
    int epochs,
    float learning_rate,
    size_t batch_size,
    float train_loss,
    float val_loss,
    float validation_split,
    int shuffle,
    unsigned int split_seed,
    int parameter_count,
    float weight_l2_norm,
    float max_abs_weight,
    float bias_l2_norm
) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"epochs\": %d,\n", epochs);
    fprintf(fp, "  \"learning_rate\": %.6f,\n", learning_rate);
    fprintf(fp, "  \"batch_size\": %zu,\n", batch_size);
    fprintf(fp, "  \"train_loss\": %.6f,\n", train_loss);
    fprintf(fp, "  \"val_loss\": %.6f,\n", val_loss);
    fprintf(fp, "  \"validation_split\": %.6f,\n", validation_split);
    fprintf(fp, "  \"shuffle\": %d,\n", shuffle);
    fprintf(fp, "  \"split_seed\": %u,\n", split_seed);
    fprintf(fp, "  \"parameter_count\": %d,\n", parameter_count);
    fprintf(fp, "  \"weight_l2_norm\": %.6f,\n", weight_l2_norm);
    fprintf(fp, "  \"max_abs_weight\": %.6f,\n", max_abs_weight);
    fprintf(fp, "  \"bias_l2_norm\": %.6f\n", bias_l2_norm);
    fprintf(fp, "}\n");

    fclose(fp);
    return 1;
}

int tinyml_write_eval_metrics_json(
    const char *path,
    float eval_loss,
    float prediction_x4,
    int parameter_count,
    float weight_l2_norm,
    float max_abs_weight,
    float bias_l2_norm
) {
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"eval_loss\": %.6f,\n", eval_loss);
    fprintf(fp, "  \"prediction_x4\": %.6f,\n", prediction_x4);
    fprintf(fp, "  \"parameter_count\": %d,\n", parameter_count);
    fprintf(fp, "  \"weight_l2_norm\": %.6f,\n", weight_l2_norm);
    fprintf(fp, "  \"max_abs_weight\": %.6f,\n", max_abs_weight);
    fprintf(fp, "  \"bias_l2_norm\": %.6f\n", bias_l2_norm);
    fprintf(fp, "}\n");

    fclose(fp);
    return 1;
}