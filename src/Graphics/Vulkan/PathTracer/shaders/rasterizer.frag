#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragTangent;
layout(location = 3) in vec3 fragBitangent;
layout(location = 4) flat in uint fragInstanceId;
layout(location = 5) flat in uint fragDiffuseIndex;
layout(location = 6) flat in uint fragNormalIndex;
layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out uint outInstanceId;

layout(binding = 2) uniform sampler2D textures[];

//layout(set = 1, binding = 0) uniform sampler2D albedoMap;

void main() {
    vec3 normal = gl_FrontFacing ? -fragNormal : fragNormal;
    vec3 texColor = texture(nonuniformEXT(textures[fragDiffuseIndex]), fragUV).rgb;
    vec3 texNormal = texture(nonuniformEXT(textures[fragNormalIndex]), fragUV).rgb;
    texNormal = normalize(texNormal * 2.0 - 1.0);
    mat3 TBN = mat3(fragTangent, fragBitangent, fragNormal);
    texNormal = TBN * texNormal;

    outNormal = vec4(normal, 1.0);
//    outNormal = vec4(texNormal, 1.0);
    //outAlbedo = vec4(normal * 0.5 + 0.5, 1.0);
    outAlbedo = vec4(texColor, 1.0);
    outInstanceId = fragInstanceId + 1;
}