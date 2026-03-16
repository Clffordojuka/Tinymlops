#include <stdio.h>
#include "tinyml.h"

int main(void) {
    printf("[train] tinyml-ops version: %s\n", tinyml_version());
    printf("[train] placeholder training entry point\n");
    return 0;
}