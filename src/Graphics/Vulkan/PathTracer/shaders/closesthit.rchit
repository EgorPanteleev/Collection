#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "utils.glsl"

struct Vertex {
    vec3 pos;
    vec2 texCoord;
    vec3 n;
    vec4 tangent;
};
struct MeshInfo {
    uint64_t vertexAddress;
    uint64_t indexAddress;
};
struct InstanceData {
    uint meshID;
    uint textureID;
    uint pad[2];
};
layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex vertices[];
};
layout(buffer_reference, scalar) readonly buffer IndexBuffer {
    uint indices[];
};
layout(binding = 0) uniform accelerationStructureEXT tlas;
layout(binding = 3) uniform DirectLight {
    vec4 dir;
    float intensity;
} directLight;
layout(binding = 4, scalar) readonly buffer MeshInfoBuffer {
    MeshInfo meshes[];
};
layout(binding = 5, scalar) readonly buffer InstanceBuffer {
    InstanceData instances[];
};
layout(binding = 6) uniform sampler2D textures[];
layout(location = 0) rayPayloadInEXT vec3 payload;
layout(location = 1) rayPayloadEXT float shadowPayload;

hitAttributeEXT vec2 hitAttrib;

void main() {
    InstanceData instance = instances[gl_InstanceCustomIndexEXT];
    MeshInfo mesh = meshes[instance.meshID];
    VertexBuffer vertexBuffer = VertexBuffer(mesh.vertexAddress);
    IndexBuffer  indexBuffer  = IndexBuffer(mesh.indexAddress);

    uint i0 = indexBuffer.indices[gl_PrimitiveID * 3 + 0];
    uint i1 = indexBuffer.indices[gl_PrimitiveID * 3 + 1];
    uint i2 = indexBuffer.indices[gl_PrimitiveID * 3 + 2];

    Vertex v0 = vertexBuffer.vertices[i0];
    Vertex v1 = vertexBuffer.vertices[i1];
    Vertex v2 = vertexBuffer.vertices[i2];

    float u = hitAttrib.x; float v = hitAttrib.y; float w = 1 - u - v;
    vec2 uv = w * v0.texCoord + u * v1.texCoord + v * v2.texCoord;

    vec3 P = w * v0.pos + u * v1.pos + v * v2.pos;
    vec3 N = normalize(w * v0.n + u * v1.n + v * v2.n);
    mat3 normalMatrix = transpose(inverse(mat3(gl_ObjectToWorldEXT)));
    P = vec3(gl_ObjectToWorldEXT * vec4(P, 1.0));
    N = normalize(normalMatrix * N);
    if (dot(N, gl_WorldRayDirectionEXT) > 0.0) N = -N;

    vec3 albedo = texture(textures[nonuniformEXT(instance.textureID)], uv).rgb;
    vec3 L = normalize(-directLight.dir.xyz);
    vec3 gN = normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos));
    gN = normalize(normalMatrix * gN);
    if (dot(gN, N) < 0.0) gN = -gN;
    float NdotL = max(dot(N, L), 0.0);
    shadowPayload = 0.0;
    traceRayEXT(
        tlas,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        ALL_OBJECTS,
        0, 1, 1,
        movedPoint(P, gN),
        T_MIN,
        L,
        T_MAX,
        1
    );

    vec3 direct = shadowPayload * albedo * NdotL * directLight.intensity;
    payload = direct;

   //payload = N * 0.5 + 0.5;//texture(textures[nonuniformEXT(instance.textureID)], uv).rgb;
}