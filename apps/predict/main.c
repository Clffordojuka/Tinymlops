#include <stdio.h>
#include "tinyml.h"

int main(void) {
    printf("[predict] tinyml-ops version: %s\n", tinyml_version());
    printf("[predict] placeholder prediction entry point\n");
    return 0;
}