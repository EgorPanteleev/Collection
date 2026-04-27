#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) flat in uint fragDiffuseIndex;
layout(location = 0) out vec4 outAlbedo;

layout(binding = 1) uniform sampler2D textures[];

//layout(set = 1, binding = 0) uniform sampler2D albedoMap;

void main() {
    vec3 texColor = texture(nonuniformEXT(textures[fragDiffuseIndex]), fragUV).rgb;
    outAlbedo = vec4(texColor, 1.0);
//    float depth = gl_FragCoord.z;
//    outAlbedo = vec4(vec3(depth), 1.0);
}