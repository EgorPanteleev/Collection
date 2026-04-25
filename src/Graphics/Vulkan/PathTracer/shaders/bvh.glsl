#include "triangle.glsl"

#define PRIM_COUNT_BITS 4
const uint MAX_PRIM = (1u << PRIM_COUNT_BITS) - 1u;

struct Node {
    BBox bbox;
    uint index;
    uint pad[3];
};

uint id(uint index) {
    return index >> PRIM_COUNT_BITS;
}

uint primCount(uint index) {
    return index & MAX_PRIM;
}

bool isLeaf(Node node) {
    return primCount(node.index) != 0;
}

struct Hit {
    uint id;
    float t, u, v;
};