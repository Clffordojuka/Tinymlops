add_library(tinyml_core
    core/tinyml.c
    core/matrix.c
    core/model.c
    core/loss.c
    core/train.c
    core/eval.c
    data/dataset.c
    io/csv.c
    io/config.c
    io/metrics.c
    io/checkpoint.c
    layers/dense.c
    utils/activation.c
)

target_include_directories(tinyml_core
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)