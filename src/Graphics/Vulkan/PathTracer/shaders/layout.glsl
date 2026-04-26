#ifndef LAYOUT_GLSL
#define LAYOUT_GLSL

#include "bvh.glsl"

layout(push_constant) uniform PushConstants {
    uint width, height, frame, maxDepth;
} pc;

layout(binding = 0) uniform Camera {
    vec4 position, forward, right, up;
    float FOV, aspectRatio, nearPlane, farPlane;
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

layout(binding = 4) buffer MaterialIndexBuffer {
    uint data[];
} materialIndexBuffer;

layout(binding = 5) uniform DirectLight {
    vec4 dir;
    float intensity;
    float pad[3];
} directLight;

layout(binding = 6, rgba8) uniform image2D image;

layout(binding = 7) uniform sampler2D textures[];

#endif
