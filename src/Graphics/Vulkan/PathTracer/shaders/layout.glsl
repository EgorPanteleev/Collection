#ifndef LAYOUT_GLSL
#define LAYOUT_GLSL

#include "bvh.glsl"

layout(push_constant) uniform PushConstants {
    uint frame, spp, minDepth, maxDepth, instanceCount;
} pc;

layout(binding = 0) uniform Camera {
    vec4 pos;
    mat4 invViewProj;
} camera;

layout(binding = 1) buffer TriangleBuffer {
    PrecomputedTriangle data[];
} triangleBuffer;

layout(binding = 2) buffer TriangleExtraBuffer {
    TriangleExtra data[];
} triangleExtraBuffer;

layout(binding = 3) buffer NodeBuffer {
    Node data[];
} nodeBuffer;

layout(binding = 4) buffer TLASNodeBuffer_ {
    Node data[];
} TLASNodeBuffer;

layout(binding = 5) buffer InstanceBuffer {
    Instance data[];
} instanceBuffer;

layout(binding = 6) uniform DirectLight {
    vec4 dir;
    float intensity;
    float pad[3];
} directLight;

layout(binding = 7, rgba8) uniform image2D outputImage;

layout(binding = 8) uniform sampler2D colorImage;
layout(binding = 9) uniform sampler2D depthImage;
layout(binding = 10) uniform sampler2D normalImage;

layout(binding = 11) uniform sampler2D textures[];

#endif
