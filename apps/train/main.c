#include <stdio.h>
#include "tinyml.h"

int main(void) {
    TinyML_Dataset dataset = tinyml_dataset_load_csv("data/samples/linear.csv");
    TinyML_DenseLayer layer = tinyml_dense_create(1, 1);

    if (dataset.sample_count == 0) {
        fprintf(stderr, "Failed to load dataset.\n");
        tinyml_dense_free(&layer);
        return 1;
    }

    tinyml_matrix_set(&layer.weights, 0, 0, 0.0f);
    tinyml_matrix_set(&layer.bias, 0, 0, 0.0f);

    const float learning_rate = 0.01f;
    const int epochs = 200;
    float final_loss = 0.0f;

    TinyML_Matrix input = tinyml_matrix_create(1, 1);
    TinyML_Matrix target = tinyml_matrix_create(1, 1);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (size_t i = 0; i < dataset.sample_count; ++i) {
            tinyml_matrix_set(&input, 0, 0, tinyml_matrix_get(&dataset.features, i, 0));
            tinyml_matrix_set(&target, 0, 0, tinyml_matrix_get(&dataset.targets, i, 0));

            epoch_loss += tinyml_train_step_dense(&layer, &input, &target, learning_rate);
        }

        epoch_loss /= (float)dataset.sample_count;
        final_loss = epoch_loss;

        if ((epoch + 1) % 20 == 0 || epoch == 0) {
            printf("Epoch %d/%d - Loss: %.6f\n", epoch + 1, epochs, epoch_loss);
        }
    }

    TinyML_Matrix test_input = tinyml_matrix_create(1, 1);
    tinyml_matrix_set(&test_input, 0, 0, 4.0f);

    TinyML_Matrix prediction = tinyml_dense_forward(&layer, &test_input);

    printf("Prediction for x=4.0: %.6f\n", tinyml_matrix_get(&prediction, 0, 0));
    printf("Learned weight: %.6f\n", tinyml_matrix_get(&layer.weights, 0, 0));
    printf("Learned bias: %.6f\n", tinyml_matrix_get(&layer.bias, 0, 0));

    if (!tinyml_write_training_metrics_json(
            "metrics/train_metrics.json",
            epochs,
            learning_rate,
            final_loss,
            tinyml_matrix_get(&layer.weights, 0, 0),
            tinyml_matrix_get(&layer.bias, 0, 0))) {
        fprintf(stderr, "Warning: failed to write metrics file.\n");
    }

    tinyml_matrix_free(&prediction);
    tinyml_matrix_free(&test_input);
    tinyml_matrix_free(&input);
    tinyml_matrix_free(&target);
    tinyml_dataset_free(&dataset);
    tinyml_dense_free(&layer);

    return 0;
}