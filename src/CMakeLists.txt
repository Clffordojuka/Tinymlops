add_library(tinyml_core
    core/tinyml.c
    core/matrix.c
    core/model.c
    core/loss.c
    layers/dense.c
    utils/activation.c
)

target_include_directories(tinyml_core
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)