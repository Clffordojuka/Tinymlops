add_executable(train_app
    train/main.c
)

target_link_libraries(train_app
    PRIVATE tinyml_core
)

add_executable(evaluate_app
    evaluate/main.c
)

target_link_libraries(evaluate_app
    PRIVATE tinyml_core
)

add_executable(predict_app
    predict/main.c
)

target_link_libraries(predict_app
    PRIVATE tinyml_core
)