#include <assert.h>
#include "tinyml.h"

int main(void) {
    size_t sizes[8];
    size_t count = 0;

    count = tinyml_parse_hidden_layers("8,4,2", sizes, 8);
    assert(count == 3);
    assert(sizes[0] == 8);
    assert(sizes[1] == 4);
    assert(sizes[2] == 2);

    count = tinyml_parse_hidden_layers("", sizes, 8);
    assert(count == 0);

    count = tinyml_parse_hidden_layers("16", sizes, 8);
    assert(count == 1);
    assert(sizes[0] == 16);

    count = tinyml_parse_hidden_layers("5,0,3", sizes, 8);
    assert(count == 2);
    assert(sizes[0] == 5);
    assert(sizes[1] == 3);

    return 0;
}