#version 460
#extension GL_EXT_ray_tracing : require

#include "utils.glsl"

layout(location = 0) rayPayloadInEXT PathPayload payload;

void main() {
    payload.radiance = vec3(0.1);
    payload.done = true;
}