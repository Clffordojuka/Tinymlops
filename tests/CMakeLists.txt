add_executable(test_smoke
    unit/test_smoke.c
)

target_link_libraries(test_smoke
    PRIVATE tinyml_core
)

add_test(NAME smoke_test COMMAND test_smoke)

add_executable(test_matrix
    unit/test_matrix.c
)

target_link_libraries(test_matrix
    PRIVATE tinyml_core
)

add_test(NAME matrix_test COMMAND test_matrix)

add_executable(test_dense
    unit/test_dense.c
)

target_link_libraries(test_dense
    PRIVATE tinyml_core
)

add_test(NAME dense_test COMMAND test_dense)