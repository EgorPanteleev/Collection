#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;
layout(location = 4) in uint inTexIndex;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragTangent;
layout(location = 3) out vec3 fragBitangent;
layout(location = 4) flat out uint fragDiffuseIndex;
layout(location = 5) flat out uint fragNormalIndex;

layout(binding = 0) uniform MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 trInvModel;
} mvp;

layout(binding = 1) readonly buffer InstanceBuffer {
    mat4 instances[];
};

void main() {
    mat4 instanceModel = instances[gl_InstanceIndex];
    mat4 worldMatrix   = mvp.model * instanceModel;
    gl_Position = mvp.proj * mvp.view * worldMatrix * vec4(inPosition, 1.0);
    mat3 normalMatrix = mat3(transpose(inverse(worldMatrix)));

    fragUV           = inUV;
    fragNormal       = normalize(normalMatrix * inNormal);
    fragTangent      = normalize(normalMatrix * inTangent.xyz);
    fragBitangent    = cross(fragNormal, fragTangent) * inTangent.w;
    fragDiffuseIndex = inTexIndex;
    fragNormalIndex  = inTexIndex + 4;
}
