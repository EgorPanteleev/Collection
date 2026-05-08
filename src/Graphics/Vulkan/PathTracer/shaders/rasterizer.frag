#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragTangent;
layout(location = 3) in vec3 fragBitangent;
layout(location = 4) flat in uint fragDiffuseIndex;
layout(location = 5) flat in uint fragNormalIndex;
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;

layout(binding = 1) uniform sampler2D textures[];

//layout(set = 1, binding = 0) uniform sampler2D albedoMap;

void main() {
    //vec3 normal = fragNormal;

    vec3 texNormal = texture(nonuniformEXT(textures[fragNormalIndex]), fragUV).rgb;
    texNormal = normalize(texNormal * 2.0 - 1.0);
    mat3 TBN = mat3(fragTangent, fragBitangent, fragNormal);
    texNormal = TBN * texNormal;

    vec3 texColor = texture(nonuniformEXT(textures[fragDiffuseIndex]), fragUV).rgb;
    outNormal = vec4(texNormal, 1.0);
    outAlbedo = vec4(texColor, 1.0);
}