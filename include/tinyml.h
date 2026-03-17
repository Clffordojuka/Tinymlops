#ifndef TINYML_H
#define TINYML_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

const char *tinyml_version(void);

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} TinyML_Matrix;

typedef struct {
    size_t input_dim;
    size_t output_dim;
    TinyML_Matrix weights;
    TinyML_Matrix bias;
} TinyML_DenseLayer;

typedef enum {
    TINYML_ACT_NONE = 0,
    TINYML_ACT_RELU = 1
} TinyML_Activation;

typedef struct {
    TinyML_DenseLayer dense;
    TinyML_Activation activation;
} TinyML_Model;

typedef struct {
    size_t sample_count;
    size_t feature_count;
    TinyML_Matrix features;
    TinyML_Matrix targets;
} TinyML_Dataset;

typedef struct {
    char data_path[256];
    int epochs;
    float learning_rate;
    char metrics_path[256];
} TinyML_TrainConfig;

/* matrix */
TinyML_Matrix tinyml_matrix_create(size_t rows, size_t cols);
void tinyml_matrix_free(TinyML_Matrix *matrix);
float tinyml_matrix_get(const TinyML_Matrix *matrix, size_t row, size_t col);
void tinyml_matrix_set(TinyML_Matrix *matrix, size_t row, size_t col, float value);
void tinyml_matrix_fill(TinyML_Matrix *matrix, float value);
TinyML_Matrix tinyml_matrix_copy(const TinyML_Matrix *matrix);
TinyML_Matrix tinyml_matrix_subtract(const TinyML_Matrix *a, const TinyML_Matrix *b);
TinyML_Matrix tinyml_matrix_transpose(const TinyML_Matrix *matrix);
TinyML_Matrix tinyml_matrix_multiply(const TinyML_Matrix *a, const TinyML_Matrix *b);
void tinyml_matrix_scale_inplace(TinyML_Matrix *matrix, float scalar);

/* dataset */
TinyML_Dataset tinyml_dataset_create(size_t sample_count, size_t feature_count);
void tinyml_dataset_free(TinyML_Dataset *dataset);
TinyML_Dataset tinyml_dataset_load_csv(const char *path);

/* config */
TinyML_TrainConfig tinyml_default_train_config(void);
int tinyml_load_train_config(const char *path, TinyML_TrainConfig *config);

/* dense layer */
TinyML_DenseLayer tinyml_dense_create(size_t input_dim, size_t output_dim);
void tinyml_dense_free(TinyML_DenseLayer *layer);
TinyML_Matrix tinyml_dense_forward(const TinyML_DenseLayer *layer, const TinyML_Matrix *input);
TinyML_Matrix tinyml_dense_backward(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *input,
    const TinyML_Matrix *grad_output,
    float learning_rate
);

/* activations */
float tinyml_relu(float x);
float tinyml_relu_derivative(float x);
void tinyml_matrix_apply_relu(TinyML_Matrix *matrix);
void tinyml_matrix_apply_relu_derivative_inplace(TinyML_Matrix *matrix);

/* loss */
float tinyml_mse_loss(const TinyML_Matrix *y_true, const TinyML_Matrix *y_pred);
TinyML_Matrix tinyml_mse_loss_gradient(const TinyML_Matrix *y_true, const TinyML_Matrix *y_pred);

/* training helpers */
float tinyml_train_step_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate
);

float tinyml_train_epoch_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *inputs,
    const TinyML_Matrix *targets,
    size_t sample_count,
    float learning_rate
);

/* metrics */
int tinyml_write_training_metrics_json(
    const char *path,
    int epochs,
    float learning_rate,
    float final_loss,
    float weight,
    float bias
);

/* model */
TinyML_Model tinyml_model_create(size_t input_dim, size_t output_dim, TinyML_Activation activation);
void tinyml_model_free(TinyML_Model *model);
TinyML_Matrix tinyml_model_forward(const TinyML_Model *model, const TinyML_Matrix *input);

#ifdef __cplusplus
}
#endif

#endif