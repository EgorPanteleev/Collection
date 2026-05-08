#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;
layout(location = 4) in uint inTexIndex;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragTangent;
layout(location = 3) out vec3 fragBitangent;
layout(location = 4) out flat uint fragDiffuseIndex;
layout(location = 5) out flat uint fragNormalIndex;

layout(binding = 0) uniform MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 trInvModel;
} mvp;

void main() {
    gl_Position = mvp.proj * mvp.view * mvp.model * vec4(inPosition, 1.0);
    fragUV = inUV;
    fragNormal = normalize(mat3(mvp.trInvModel) * inNormal);
    fragTangent = normalize(mat3(mvp.trInvModel) * inTangent.xyz);
    fragBitangent = cross(fragNormal, fragTangent) * inTangent.w;
    fragDiffuseIndex = inTexIndex;
    fragNormalIndex = inTexIndex + 4;
}
