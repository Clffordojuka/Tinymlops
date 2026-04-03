#include <string.h>
#include "tinyml.h"

static int tinyml_config_is_mlp(const TinyML_TrainConfig *config)
{
    return strcmp(config->model_type, "mlp") == 0;
}

static int tinyml_config_is_deep_mlp(const TinyML_TrainConfig *config)
{
    return strcmp(config->model_type, "deep_mlp") == 0;
}

int tinyml_runtime_model_init(
    TinyML_RuntimeModel *model,
    const TinyML_TrainConfig *config,
    size_t input_dim)
{
    if (model == NULL || config == NULL)
    {
        return 0;
    }

    if (tinyml_config_is_deep_mlp(config))
    {
        size_t hidden_sizes[16];
        size_t hidden_count = tinyml_parse_hidden_layers(
            config->hidden_layers,
            hidden_sizes,
            16);
        TinyML_Activation hidden_activation = tinyml_activation_from_string(config->hidden_activation);

        if (hidden_count == 0)
        {
            return 0;
        }

        model->kind = TINYML_MODEL_DEEP_MLP;
        model->deep_mlp = tinyml_deep_mlp_create(
            input_dim,
            hidden_sizes,
            hidden_count,
            1,
            hidden_activation);
        return model->deep_mlp.layers != NULL;
    }

    if (tinyml_config_is_mlp(config))
    {
        size_t hidden_dim = config->hidden_dim > 0 ? config->hidden_dim : 8;
        TinyML_Activation hidden_activation = tinyml_activation_from_string(config->hidden_activation);
        model->kind = TINYML_MODEL_MLP;
        model->mlp = tinyml_mlp_create(input_dim, hidden_dim, 1, hidden_activation);
    }
    else
    {
        model->kind = TINYML_MODEL_LINEAR;
        model->linear = tinyml_dense_create(input_dim, 1);
    }

    return 1;
}

void tinyml_runtime_model_free(TinyML_RuntimeModel *model)
{
    if (model == NULL)
    {
        return;
    }

    if (model->kind == TINYML_MODEL_MLP)
    {
        tinyml_mlp_free(&model->mlp);
    }
    else if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        tinyml_deep_mlp_free(&model->deep_mlp);
    }
    else
    {
        tinyml_dense_free(&model->linear);
    }
}

TinyML_Matrix tinyml_runtime_model_forward(
    const TinyML_RuntimeModel *model,
    const TinyML_Matrix *input)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_mlp_forward(&model->mlp, input);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        return tinyml_deep_mlp_forward(&model->deep_mlp, input);
    }

    return tinyml_dense_forward(&model->linear, input);
}

float tinyml_runtime_model_train_step(
    TinyML_RuntimeModel *model,
    const TinyML_Matrix *input,
    const TinyML_Matrix *target,
    float learning_rate,
    float l2_lambda,
    TinyML_OptimizerType optimizer,
    float adam_beta1,
    float adam_beta2,
    float adam_epsilon)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_train_step_mlp(
            &model->mlp,
            input,
            target,
            learning_rate,
            l2_lambda,
            optimizer,
            adam_beta1,
            adam_beta2,
            adam_epsilon);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        return tinyml_train_step_deep_mlp(
            &model->deep_mlp,
            input,
            target,
            learning_rate,
            l2_lambda,
            optimizer,
            adam_beta1,
            adam_beta2,
            adam_epsilon);
    }

    return tinyml_train_step_dense(
        &model->linear,
        input,
        target,
        learning_rate,
        l2_lambda,
        optimizer,
        adam_beta1,
        adam_beta2,
        adam_epsilon);
}

float tinyml_runtime_model_evaluate(
    const TinyML_RuntimeModel *model,
    const TinyML_Dataset *dataset)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_evaluate_mlp(&model->mlp, dataset);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        return tinyml_evaluate_deep_mlp(&model->deep_mlp, dataset);
    }

    return tinyml_evaluate_dense(&model->linear, dataset);
}

float tinyml_runtime_model_predict_single(
    const TinyML_RuntimeModel *model,
    float x)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_predict_mlp_single(&model->mlp, x);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        return tinyml_predict_deep_mlp_single(&model->deep_mlp, x);
    }

    return tinyml_predict_dense_single(&model->linear, x);
}

int tinyml_runtime_model_save_checkpoint(
    const TinyML_RuntimeModel *model,
    const char *path)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_save_mlp_checkpoint(path, &model->mlp);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        return tinyml_save_deep_mlp_checkpoint(path, &model->deep_mlp);
    }

    return tinyml_save_dense_checkpoint(path, &model->linear);
}

int tinyml_runtime_model_load_checkpoint(
    TinyML_RuntimeModel *model,
    const TinyML_TrainConfig *config,
    const char *path)
{
    if (model == NULL || config == NULL)
    {
        return 0;
    }

    if (tinyml_config_is_deep_mlp(config))
    {
        model->kind = TINYML_MODEL_DEEP_MLP;
        return tinyml_load_deep_mlp_checkpoint(path, &model->deep_mlp);
    }

    if (tinyml_config_is_mlp(config))
    {
        model->kind = TINYML_MODEL_MLP;
        return tinyml_load_mlp_checkpoint(path, &model->mlp);
    }

    model->kind = TINYML_MODEL_LINEAR;
    return tinyml_load_dense_checkpoint(path, &model->linear);
}

int tinyml_runtime_model_parameter_count(const TinyML_RuntimeModel *model)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_dense_parameter_count(&model->mlp.hidden) +
               tinyml_dense_parameter_count(&model->mlp.output);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        int total = 0;
        for (size_t i = 0; i < model->deep_mlp.num_layers; ++i)
        {
            total += tinyml_dense_parameter_count(&model->deep_mlp.layers[i]);
        }
        return total;
    }

    return tinyml_dense_parameter_count(&model->linear);
}

float tinyml_runtime_model_weight_l2_norm(const TinyML_RuntimeModel *model)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_dense_weight_l2_norm(&model->mlp.hidden) +
               tinyml_dense_weight_l2_norm(&model->mlp.output);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        float total = 0.0f;
        for (size_t i = 0; i < model->deep_mlp.num_layers; ++i)
        {
            total += tinyml_dense_weight_l2_norm(&model->deep_mlp.layers[i]);
        }
        return total;
    }

    return tinyml_dense_weight_l2_norm(&model->linear);
}

float tinyml_runtime_model_max_abs_weight(const TinyML_RuntimeModel *model)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        float a = tinyml_dense_max_abs_weight(&model->mlp.hidden);
        float b = tinyml_dense_max_abs_weight(&model->mlp.output);
        return a > b ? a : b;
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        float max_value = 0.0f;
        for (size_t i = 0; i < model->deep_mlp.num_layers; ++i)
        {
            float v = tinyml_dense_max_abs_weight(&model->deep_mlp.layers[i]);
            if (v > max_value)
            {
                max_value = v;
            }
        }
        return max_value;
    }

    return tinyml_dense_max_abs_weight(&model->linear);
}

float tinyml_runtime_model_bias_l2_norm(const TinyML_RuntimeModel *model)
{
    if (model->kind == TINYML_MODEL_MLP)
    {
        return tinyml_dense_bias_l2_norm(&model->mlp.hidden) +
               tinyml_dense_bias_l2_norm(&model->mlp.output);
    }
    if (model->kind == TINYML_MODEL_DEEP_MLP)
    {
        float total = 0.0f;
        for (size_t i = 0; i < model->deep_mlp.num_layers; ++i)
        {
            total += tinyml_dense_bias_l2_norm(&model->deep_mlp.layers[i]);
        }
        return total;
    }

    return tinyml_dense_bias_l2_norm(&model->linear);
}