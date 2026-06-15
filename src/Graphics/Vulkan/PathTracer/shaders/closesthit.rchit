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
layout(binding = 4, scalar) uniform DirectLight {
    vec3 dir;
    float intensity;
} directLight;
layout(binding = 5, scalar) readonly buffer MeshInfoBuffer {
    MeshInfo meshes[];
};
layout(binding = 6, scalar) readonly buffer InstanceBuffer {
    InstanceData instances[];
};
layout(binding = 7, scalar) readonly buffer EmissiveIndexBuffer {
    uint emissiveIndices[];
};
layout(binding = 8, scalar) readonly buffer MaterialBuffer {
    MaterialData materials[];
};
layout(binding = 9) uniform sampler2D textures[];
layout(location = 0) rayPayloadInEXT PathPayload payload;
layout(location = 1) rayPayloadEXT float shadowPayload;

hitAttributeEXT vec2 hitAttrib;

vec3 getBaseColor(MaterialData material, vec2 uv) {
    vec3 baseColor = material.baseColor.xyz;
    if (material.baseColorTexIndex != UINT_MAX)
        baseColor *= texture(textures[nonuniformEXT(material.baseColorTexIndex)], uv).rgb;
    return baseColor;
}

float shadow(vec3 P, vec3 N, vec3 L, float t_max) {
    shadowPayload = 0.0;
    traceRayEXT(
        tlas,
        gl_RayFlagsTerminateOnFirstHitEXT |
        gl_RayFlagsSkipClosestHitShaderEXT,
        ALL_OBJECTS,
        0, 1, 1,
        movedPoint(P, N),
        T_MIN,
        L,
        t_max - T_MIN,
        1
    );
    return shadowPayload;
}

struct LightSample {
    vec3  P;
    vec3  N;
    vec3  emission;
    float pdf;
};

LightSample sampleEmissiveMesh(inout uint seed, uint instanceIndex) {
    InstanceData instance = instances[instanceIndex];
    MeshInfo     mesh     = meshes[instance.meshIndex];
    MaterialData material = materials[instance.materialIndex];

    VertexBuffer vertexBuffer = VertexBuffer(mesh.vertexAddress);
    IndexBuffer  indexBuffer = IndexBuffer(mesh.indexAddress);

    uint triCount = mesh.indexCount / 3;
    uint triIndex = uint(rand01(seed) * float(triCount));

    Vertex v0 = vertexBuffer.vertices[indexBuffer.indices[triIndex * 3 + 0]];
    Vertex v1 = vertexBuffer.vertices[indexBuffer.indices[triIndex * 3 + 1]];
    Vertex v2 = vertexBuffer.vertices[indexBuffer.indices[triIndex * 3 + 2]];

    float r1 = sqrt(rand01(seed));
    float r2 = rand01(seed);
    float bu = 1.0 - r1;
    float bv = r1 * (1.0 - r2);
    float bw = r1 * r2;

    mat4 xform = instance.model;
    vec3 lP = bu * v0.pos + bv * v1.pos + bw * v2.pos;
    vec3 lN = normalize(bu * v0.n  + bv * v1.n  + bw * v2.n);

    lP = vec3(xform * vec4(lP, 1.0));
    mat3 nm = transpose(inverse(mat3(xform)));
    lN = normalize(nm * lN);

    vec3 e1 = vec3(xform * vec4(v1.pos - v0.pos, 0.0));
    vec3 e2 = vec3(xform * vec4(v2.pos - v0.pos, 0.0));
    float triArea = 0.5 * length(cross(e1, e2));

    LightSample ls;
    ls.P        = lP;
    ls.N        = lN;
    ls.emission = material.emissionColor * material.luminance;
    ls.pdf      = 1.0 / (float(triCount) * triArea);
    return ls;
}

void main() {
    payload.instanceId = gl_InstanceCustomIndexEXT;
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
    if (material.luminance > 0.0 && payload.depth == 0) {
        payload.radiance += payload.throughput * material.emissionColor * material.luminance;
        payload.done = true;
        return;
    }

    vec3 L = normalize(-directLight.dir.xyz);
    vec3 gN = normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos));
    gN = normalize(normalMatrix * gN);
    if (dot(gN, N) < 0.0) gN = -gN;
    vec3 wo = -gl_WorldRayDirectionEXT;
    Scatter scatter = lambertianScatter(payload.seed, baseColor, N, wo);

    if (directLight.intensity > 0.0) {
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float vis = shadow(P, gN, L, T_MAX);
            vec3 direct = scatter.brdf * NdotL * directLight.intensity * vis;
            payload.radiance += payload.throughput * direct;
        }
    }

    // --- Emissive mesh NEE ---
    uint emissiveCount = uint(emissiveIndices.length());
    uint lightIdx       = uint(rand01(payload.seed) * float(emissiveCount));
    uint instanceIndex  = emissiveIndices[lightIdx];
    if (instanceIndex != UINT_MAX) {
        LightSample ls  = sampleEmissiveMesh(payload.seed, instanceIndex);

        vec3  toLight   = ls.P - P;
        float distSq    = dot(toLight, toLight);
        float dist      = sqrt(distSq);
        vec3  wi        = toLight / dist;

        float NdotWi    = max(dot(N,     wi),  0.0);
        float LNdotWi   = max(dot(ls.N, -wi),  0.0); // light's normal facing us

        if (NdotWi > 0.0 && LNdotWi > 0.0) {
            float vis1 = shadow(P, gN, wi, dist * (1.0 - SHADOW_EPS));
            float pdfSolidAngle = ls.pdf * distSq / LNdotWi;
            pdfSolidAngle /= float(emissiveCount);
            vec3 brdf = scatter.brdf;
            payload.radiance += payload.throughput
            * brdf * ls.emission * NdotWi * vis1
            / pdfSolidAngle;
        }
    }

//    vec3 L = normalize(-directLight.dir.xyz);
//    vec3 gN = normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos));
//    gN = normalize(normalMatrix * gN);
//    if (dot(gN, N) < 0.0) gN = -gN;
//    float NdotL = max(dot(N, L), 0.0);
//    shadowPayload = shadow(P, gN, L);
//
//    vec3 wo = -gl_WorldRayDirectionEXT;
//    Scatter scatter = lambertianScatter(payload.seed, baseColor, N, wo);
//    vec3 direct  = scatter.brdf * NdotL * directLight.intensity * shadowPayload;
//    payload.radiance += payload.throughput * direct;

    payload.throughput *= scatter.brdf * max(dot(scatter.wi, N), 0.0) / scatter.pdf;
    payload.origin    = movedPoint(P, gN);
    payload.direction = scatter.wi;
    payload.done      = false;
}