#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) out uint outInstance;

void main() {
    outInstance = 1u;
}