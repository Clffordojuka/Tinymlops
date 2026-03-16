#include <stdio.h>
#include "tinyml.h"

int main(void) {
    printf("[evaluate] tinyml-ops version: %s\n", tinyml_version());
    printf("[evaluate] placeholder evaluation entry point\n");
    return 0;
}