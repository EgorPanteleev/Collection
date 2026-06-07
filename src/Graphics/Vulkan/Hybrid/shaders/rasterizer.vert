#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragTangent;
layout(location = 3) out vec3 fragBitangent;
layout(location = 4) flat out uint fragInstanceId;
layout(location = 5) flat out uint fragDiffuseIndex;
layout(location = 6) flat out uint fragNormalIndex;

layout(binding = 0) uniform MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 trInvModel;
} mvp;

struct Instance {
    mat4 model;
    mat4 invModel;
    uint texIndex;
};

layout(binding = 1) readonly buffer InstanceBuffer {
    Instance instances[];
};

void main() {
    Instance instance = instances[gl_InstanceIndex];
    gl_Position = mvp.proj * mvp.view * instance.model * vec4(inPosition, 1.0);
    mat3 normalMatrix = mat3(transpose(instance.invModel));

    fragUV           = inUV;
    fragNormal       = normalize(normalMatrix * inNormal);
    fragTangent      = normalize(normalMatrix * inTangent.xyz);
    fragTangent = normalize(fragTangent - fragNormal * dot(fragNormal, fragTangent));
    fragBitangent    = (cross(fragNormal, fragTangent) * inTangent.w);

    fragInstanceId   = gl_InstanceIndex;
    fragDiffuseIndex = instance.texIndex;
    fragNormalIndex  = instance.texIndex + 4;
}
