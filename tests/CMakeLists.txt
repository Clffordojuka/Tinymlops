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