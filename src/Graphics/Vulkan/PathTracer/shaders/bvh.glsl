#ifndef BVH_GLSL
#define BVH_GLSL

#include "triangle.glsl"

#define PRIM_COUNT_BITS 3
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

struct BLASHit {
    uint id;
    float t, u, v;
};

struct Hit {
    uint triId;
    uint instanceId;
    float t, u, v;
};

struct Instance {
    mat4 model;
    mat4 invModel;
    uint baseNode;
    uint baseTri;
    float texIndex;
};

#endif