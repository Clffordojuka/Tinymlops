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

/* matrix */
TinyML_Matrix tinyml_matrix_create(size_t rows, size_t cols);
void tinyml_matrix_free(TinyML_Matrix *matrix);
float tinyml_matrix_get(const TinyML_Matrix *matrix, size_t row, size_t col);
void tinyml_matrix_set(TinyML_Matrix *matrix, size_t row, size_t col, float value);

/* dense layer */
TinyML_DenseLayer tinyml_dense_create(size_t input_dim, size_t output_dim);
void tinyml_dense_free(TinyML_DenseLayer *layer);
TinyML_Matrix tinyml_dense_forward(const TinyML_DenseLayer *layer, const TinyML_Matrix *input);

/* activations */
float tinyml_relu(float x);

#ifdef __cplusplus
}
#endif

#endif