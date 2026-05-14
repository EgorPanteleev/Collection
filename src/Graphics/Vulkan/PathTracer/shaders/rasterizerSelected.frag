#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragTangent;
layout(location = 3) in vec3 fragBitangent;
layout(location = 4) flat in uint fragInstanceId;
layout(location = 5) flat in uint fragDiffuseIndex;
layout(location = 6) flat in uint fragNormalIndex;

layout(location = 0) out uint outSelectedInstance;

void main() {
    outSelectedInstance = 1u;
}