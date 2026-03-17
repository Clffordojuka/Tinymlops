#include <assert.h>
#include <string.h>
#include "tinyml.h"

#ifndef TINYML_SOURCE_DIR
#define TINYML_SOURCE_DIR "."
#endif

int main(void) {
    TinyML_TrainConfig config;
    const char *cfg_path = TINYML_SOURCE_DIR "/configs/base/train_linear.cfg";

    int ok = tinyml_load_train_config(cfg_path, &config);
    assert(ok == 1);

    assert(strcmp(config.data_path, "data/samples/linear.csv") == 0);
    assert(config.epochs == 200);
    assert(config.learning_rate == 0.01f);
    assert(strcmp(config.metrics_path, "metrics/train_metrics.json") == 0);

    return 0;
}