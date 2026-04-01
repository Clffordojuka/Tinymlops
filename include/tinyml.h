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
    TINYML_ACT_RELU = 1,
    TINYML_ACT_TANH = 2,
    TINYML_ACT_LINEAR = 3
} TinyML_Activation;

typedef struct {
    TinyML_DenseLayer dense;
    TinyML_Activation activation;
} TinyML_Model;

typedef struct {
    TinyML_DenseLayer hidden;
    TinyML_DenseLayer output;
    TinyML_Activation hidden_activation;
} TinyML_MLP;

typedef struct {
    size_t num_layers;
    TinyML_DenseLayer *layers;
    TinyML_Activation hidden_activation;
} TinyML_DeepMLP;

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
    char lr_schedule[32];
    int lr_step_size;
    float lr_decay;
    float l2_lambda;
    char model_type[32];
    size_t hidden_dim;
    char hidden_layers[128];
    char hidden_activation[32];
    char metrics_path[256];
    char checkpoint_path[256];
    float validation_split;
    float test_split;
    int shuffle;
    unsigned int split_seed;
    char eval_metrics_path[256];
    char normalization_path[256];
    size_t batch_size;
    int patience;
    float min_delta;
    int save_best_only;
} TinyML_TrainConfig;

typedef struct {
    size_t feature_count;
    float *mean;
    float *std;
} TinyML_NormalizationStats;

typedef enum {
    TINYML_MODEL_LINEAR = 0,
    TINYML_MODEL_MLP = 1,
    TINYML_MODEL_DEEP_MLP = 2
} TinyML_ModelKind;

typedef struct {
    TinyML_ModelKind kind;
    TinyML_DenseLayer linear;
    TinyML_MLP mlp;
    TinyML_DeepMLP deep_mlp;
} TinyML_RuntimeModel;

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
void tinyml_dataset_split(
    const TinyML_Dataset *dataset,
    float validation_split,
    int shuffle,
    unsigned int split_seed,
    TinyML_Dataset *train_dataset,
    TinyML_Dataset *val_dataset
);
void tinyml_dataset_split_three_way(
    const TinyML_Dataset *dataset,
    float validation_split,
    float test_split,
    int shuffle,
    unsigned int split_seed,
    TinyML_Dataset *train_dataset,
    TinyML_Dataset *val_dataset,
    TinyML_Dataset *test_dataset
);

/* config */
TinyML_TrainConfig tinyml_default_train_config(void);
int tinyml_load_train_config(const char *path, TinyML_TrainConfig *config);
size_t tinyml_parse_hidden_layers(const char *text, size_t *sizes, size_t max_sizes);

/* dense layer */
TinyML_DenseLayer tinyml_dense_create(size_t input_dim, size_t output_dim);
void tinyml_dense_free(TinyML_DenseLayer *layer);
TinyML_Matrix tinyml_dense_forward(const TinyML_DenseLayer *layer, const TinyML_Matrix *input);
TinyML_Matrix tinyml_dense_backward(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *input,
    const TinyML_Matrix *grad_output,
    float learning_rate,
    float l2_lambda
);

/* activations */
float tinyml_relu(float x);
float tinyml_relu_derivative(float x);
float tinyml_tanh(float x);
float tinyml_tanh_derivative(float x);

void tinyml_matrix_apply_relu(TinyML_Matrix *matrix);
void tinyml_matrix_apply_relu_derivative_inplace(TinyML_Matrix *matrix);
void tinyml_matrix_apply_tanh(TinyML_Matrix *matrix);
void tinyml_matrix_apply_tanh_derivative_inplace(TinyML_Matrix *matrix);

TinyML_Activation tinyml_activation_from_string(const char *name);
const char *tinyml_activation_to_string(TinyML_Activation activation);
void tinyml_matrix_apply_activation(TinyML_Matrix *matrix, TinyML_Activation activation);
void tinyml_matrix_apply_activation_derivative_inplace(TinyML_Matrix *matrix, TinyML_Activation activation);

/* loss */
float tinyml_mse_loss(const TinyML_Matrix *y_true, const TinyML_Matrix *y_pred);
TinyML_Matrix tinyml_mse_loss_gradient(const TinyML_Matrix *y_true, const TinyML_Matrix *y_pred);

/* training helpers */
float tinyml_train_step_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda
);

float tinyml_train_batch_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *inputs,
    const TinyML_Matrix *targets,
    float learning_rate,
    float l2_lambda
);

float tinyml_train_epoch_dense(
    TinyML_DenseLayer *layer,
    const TinyML_Matrix *inputs,
    const TinyML_Matrix *targets,
    size_t sample_count,
    float learning_rate,
    float l2_lambda
);

/* metrics */
int tinyml_dense_parameter_count(const TinyML_DenseLayer *layer);
float tinyml_dense_weight_l2_norm(const TinyML_DenseLayer *layer);
float tinyml_dense_max_abs_weight(const TinyML_DenseLayer *layer);
float tinyml_dense_bias_l2_norm(const TinyML_DenseLayer *layer);

int tinyml_write_training_metrics_json(
    const char *path,
    int epochs,
    float learning_rate,
    float final_learning_rate,
    const char *lr_schedule,
    int lr_step_size,
    float lr_decay,
    float l2_lambda,
    const char *model_type,
    size_t hidden_dim,
    const char *hidden_activation,
    size_t batch_size,
    float train_loss,
    float val_loss,
    float validation_split,
    int shuffle,
    unsigned int split_seed,
    int parameter_count,
    float weight_l2_norm,
    float max_abs_weight,
    float bias_l2_norm,
    float best_val_loss,
    int best_epoch,
    int stopped_early,
    int patience,
    float min_delta,
    int save_best_only
);

int tinyml_write_eval_metrics_json(
    const char *path,
    float eval_loss,
    float prediction_x4,
    int parameter_count,
    float weight_l2_norm,
    float max_abs_weight,
    float bias_l2_norm
);

/* checkpoint */
int tinyml_save_dense_checkpoint(const char *path, const TinyML_DenseLayer *layer);
int tinyml_load_dense_checkpoint(const char *path, TinyML_DenseLayer *layer);
int tinyml_save_mlp_checkpoint(const char *path, const TinyML_MLP *mlp);
int tinyml_load_mlp_checkpoint(const char *path, TinyML_MLP *mlp);
int tinyml_save_deep_mlp_checkpoint(const char *path, const TinyML_DeepMLP *mlp);
int tinyml_load_deep_mlp_checkpoint(const char *path, TinyML_DeepMLP *mlp);

/* evaluation */
float tinyml_evaluate_dense(
    const TinyML_DenseLayer *layer,
    const TinyML_Dataset *dataset
);
float tinyml_predict_dense_single(const TinyML_DenseLayer *layer, float x);

/* model */
TinyML_Model tinyml_model_create(size_t input_dim, size_t output_dim, TinyML_Activation activation);
void tinyml_model_free(TinyML_Model *model);
TinyML_Matrix tinyml_model_forward(const TinyML_Model *model, const TinyML_Matrix *input);

TinyML_MLP tinyml_mlp_create(
    size_t input_dim,
    size_t hidden_dim,
    size_t output_dim,
    TinyML_Activation hidden_activation
);
void tinyml_mlp_free(TinyML_MLP *mlp);
TinyML_Matrix tinyml_mlp_forward(const TinyML_MLP *mlp, const TinyML_Matrix *input);

TinyML_DeepMLP tinyml_deep_mlp_create(
    size_t input_dim,
    const size_t *hidden_sizes,
    size_t hidden_count,
    size_t output_dim,
    TinyML_Activation hidden_activation
);
void tinyml_deep_mlp_free(TinyML_DeepMLP *mlp);
TinyML_Matrix tinyml_deep_mlp_forward(const TinyML_DeepMLP *mlp, const TinyML_Matrix *input);

float tinyml_train_step_mlp(
    TinyML_MLP *mlp,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda
);

float tinyml_train_step_deep_mlp(
    TinyML_DeepMLP *mlp,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda
);

float tinyml_evaluate_mlp(
    const TinyML_MLP *mlp,
    const TinyML_Dataset *dataset
);

float tinyml_evaluate_deep_mlp(
    const TinyML_DeepMLP *mlp,
    const TinyML_Dataset *dataset
);

float tinyml_predict_mlp_single(const TinyML_MLP *mlp, float x);
float tinyml_predict_deep_mlp_single(const TinyML_DeepMLP *mlp, float x);

/* normalization */
TinyML_NormalizationStats tinyml_normalization_stats_create(size_t feature_count);
void tinyml_normalization_stats_free(TinyML_NormalizationStats *stats);

TinyML_NormalizationStats tinyml_fit_normalization(const TinyML_Dataset *dataset);
void tinyml_apply_normalization(TinyML_Dataset *dataset, const TinyML_NormalizationStats *stats);
float tinyml_normalize_single_value(float value, float mean, float std);

int tinyml_save_normalization_stats(const char *path, const TinyML_NormalizationStats *stats);
int tinyml_load_normalization_stats(const char *path, TinyML_NormalizationStats *stats);

/* runtime model */
int tinyml_runtime_model_init(
    TinyML_RuntimeModel *model,
    const TinyML_TrainConfig *config,
    size_t input_dim
);

void tinyml_runtime_model_free(TinyML_RuntimeModel *model);

TinyML_Matrix tinyml_runtime_model_forward(
    const TinyML_RuntimeModel *model,
    const TinyML_Matrix *input
);

float tinyml_runtime_model_train_step(
    TinyML_RuntimeModel *model,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda
);

float tinyml_runtime_model_evaluate(
    const TinyML_RuntimeModel *model,
    const TinyML_Dataset *dataset
);

float tinyml_runtime_model_predict_single(
    const TinyML_RuntimeModel *model,
    float x
);

int tinyml_runtime_model_save_checkpoint(
    const TinyML_RuntimeModel *model,
    const char *path
);

int tinyml_runtime_model_load_checkpoint(
    TinyML_RuntimeModel *model,
    const TinyML_TrainConfig *config,
    const char *path
);

int tinyml_runtime_model_parameter_count(const TinyML_RuntimeModel *model);
float tinyml_runtime_model_weight_l2_norm(const TinyML_RuntimeModel *model);
float tinyml_runtime_model_max_abs_weight(const TinyML_RuntimeModel *model);
float tinyml_runtime_model_bias_l2_norm(const TinyML_RuntimeModel *model);

#ifdef __cplusplus
}
#endif

#endif