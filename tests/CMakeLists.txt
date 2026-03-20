add_executable(test_smoke
    unit/test_smoke.c
)
target_link_libraries(test_smoke PRIVATE tinyml_core)
add_test(NAME smoke_test COMMAND test_smoke)

add_executable(test_matrix
    unit/test_matrix.c
)
target_link_libraries(test_matrix PRIVATE tinyml_core)
add_test(NAME matrix_test COMMAND test_matrix)

add_executable(test_dense
    unit/test_dense.c
)
target_link_libraries(test_dense PRIVATE tinyml_core)
add_test(NAME dense_test COMMAND test_dense)

add_executable(test_activation
    unit/test_activation.c
)
target_link_libraries(test_activation PRIVATE tinyml_core)
add_test(NAME activation_test COMMAND test_activation)

add_executable(test_loss
    unit/test_loss.c
)
target_link_libraries(test_loss PRIVATE tinyml_core)
add_test(NAME loss_test COMMAND test_loss)

add_executable(test_model
    unit/test_model.c
)
target_link_libraries(test_model PRIVATE tinyml_core)
add_test(NAME model_test COMMAND test_model)

add_executable(test_training_step
    unit/test_training_step.c
)
target_link_libraries(test_training_step PRIVATE tinyml_core)
add_test(NAME training_step_test COMMAND test_training_step)

add_executable(test_linear_fit
    integration/test_linear_fit.c
)
target_link_libraries(test_linear_fit PRIVATE tinyml_core)
add_test(NAME linear_fit_test COMMAND test_linear_fit)

add_executable(test_dataset
    unit/test_dataset.c
)
target_link_libraries(test_dataset PRIVATE tinyml_core)
target_compile_definitions(test_dataset PRIVATE TINYML_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME dataset_test COMMAND test_dataset)

add_executable(test_config
    unit/test_config.c
)
target_link_libraries(test_config PRIVATE tinyml_core)
target_compile_definitions(test_config PRIVATE TINYML_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME config_test COMMAND test_config)

add_executable(test_checkpoint
    unit/test_checkpoint.c
)
target_link_libraries(test_checkpoint PRIVATE tinyml_core)
add_test(NAME checkpoint_test COMMAND test_checkpoint)

add_executable(test_evaluate
    integration/test_evaluate.c
)
target_link_libraries(test_evaluate PRIVATE tinyml_core)
target_compile_definitions(test_evaluate PRIVATE TINYML_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME evaluate_test COMMAND test_evaluate)

add_executable(test_predict
    integration/test_predict.c
)
target_link_libraries(test_predict PRIVATE tinyml_core)
target_compile_definitions(test_predict PRIVATE TINYML_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME predict_test COMMAND test_predict)

add_executable(test_dataset_split
    unit/test_dataset_split.c
)
target_link_libraries(test_dataset_split PRIVATE tinyml_core)
add_test(NAME dataset_split_test COMMAND test_dataset_split)

add_executable(test_normalization
    unit/test_normalization.c
)
target_link_libraries(test_normalization PRIVATE tinyml_core)
add_test(NAME normalization_test COMMAND test_normalization)