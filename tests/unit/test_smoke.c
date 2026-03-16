#include <assert.h>
#include <string.h>
#include "tinyml.h"

int main(void) {
    const char *version = tinyml_version();

    assert(version != NULL);
    assert(strlen(version) > 0);
    assert(strcmp(version, "0.1.0") == 0);

    return 0;
}