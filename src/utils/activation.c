#include "tinyml.h"

float tinyml_relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}