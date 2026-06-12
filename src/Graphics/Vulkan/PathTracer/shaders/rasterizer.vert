#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

layout(binding = 0) uniform MVPBuffer {
    mat4 model;
    mat4 view;
    mat4 proj;
} mvp;

layout(binding = 1) uniform InstanceBuffer {
    mat4 model;
} instance;


void main() {
    gl_Position = mvp.proj * mvp.view * instance.model * vec4(inPosition, 1.0);
}
