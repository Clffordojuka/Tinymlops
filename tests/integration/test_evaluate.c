#include <assert.h>
#include "tinyml.h"

#ifndef TINYML_SOURCE_DIR
#define TINYML_SOURCE_DIR "."
#endif

int main(void) {
    const char *csv_path = TINYML_SOURCE_DIR "/data/samples/linear.csv";
    const char *ckpt_path = "test_eval_model.txt";

    TinyML_Dataset dataset = tinyml_dataset_load_csv(csv_path);
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);

    tinyml_matrix_set(&layer.weights, 0, 0, 2.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    assert(tinyml_save_dense_checkpoint(ckpt_path, &layer) == 1);

    TinyML_DenseLayer loaded;
    assert(tinyml_load_dense_checkpoint(ckpt_path, &loaded) == 1);

    float eval_loss = tinyml_evaluate_dense(&loaded, &dataset);
    assert(eval_loss < 0.0001f);

    tinyml_dense_free(&layer);
    tinyml_dense_free(&loaded);
    tinyml_dataset_free(&dataset);

    return 0;
}