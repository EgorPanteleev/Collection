#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragTangent;
layout(location = 3) flat in uint fragDiffuseIndex;
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;

layout(binding = 1) uniform sampler2D textures[];

//layout(set = 1, binding = 0) uniform sampler2D albedoMap;

void main() {
    vec3 worldNormal = normalize(fragNormal);
    vec3 texColor = texture(nonuniformEXT(textures[fragDiffuseIndex]), fragUV).rgb;
    outNormal = vec4(worldNormal * 0.5 + 0.5, 1.0);
    outAlbedo = vec4(texColor, 1.0);
}