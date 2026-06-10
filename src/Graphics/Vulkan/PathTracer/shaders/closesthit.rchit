#version 460

#include "utils.glsl"
#include "material.glsl"

layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex vertices[];
};
layout(buffer_reference, scalar) readonly buffer IndexBuffer {
    uint indices[];
};
layout(binding = 0) uniform accelerationStructureEXT tlas;
layout(binding = 4) uniform DirectLight {
    vec4 dir;
    float intensity;
} directLight;
layout(binding = 5, scalar) readonly buffer MeshInfoBuffer {
    MeshInfo meshes[];
};
layout(binding = 6, scalar) readonly buffer InstanceBuffer {
    InstanceData instances[];
};
layout(binding = 7) readonly buffer MaterialBuffer {
    MaterialData materials[];
};
layout(binding = 8) uniform sampler2D textures[];
layout(location = 0) rayPayloadInEXT PathPayload payload;
layout(location = 1) rayPayloadEXT float shadowPayload;

hitAttributeEXT vec2 hitAttrib;

vec3 getBaseColor(MaterialData material, vec2 uv) {
    vec3 baseColor = material.baseColor.xyz;
    if (material.baseColorTexIndex != UINT_MAX)
        baseColor *= texture(textures[nonuniformEXT(material.baseColorTexIndex)], uv).rgb;
    return baseColor;
}

void main() {
    InstanceData instance = instances[gl_InstanceCustomIndexEXT];
    MeshInfo mesh = meshes[instance.meshIndex];
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

    MaterialData material = materials[instance.materialIndex];
    vec3 baseColor = getBaseColor(material, uv);
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

    vec3 wo = -gl_WorldRayDirectionEXT;
    Scatter scatter = lambertianScatter(payload.seed, baseColor, N, wo);
    vec3 direct  = scatter.brdf * NdotL * directLight.intensity * shadowPayload;
    payload.radiance += payload.throughput * direct;
    payload.throughput *= scatter.brdf * max(dot(scatter.wi, N), 0.0) / scatter.pdf;

    payload.origin    = movedPoint(P, gN);
    payload.direction = scatter.wi;
    payload.instanceId = gl_InstanceCustomIndexEXT;
    payload.done      = false;
}