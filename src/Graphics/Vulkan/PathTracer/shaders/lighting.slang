module lighting;

import common;
import types;
import scene;
import bsdf;

public float traceShadow(float3 P, float3 N, float3 L, float tMax) {
    float visible = 0.0;
    RayDesc ray;
    ray.Origin    = movedPoint(P, N);
    ray.TMin      = T_MIN;
    ray.Direction = L;
    ray.TMax      = tMax - T_MIN;
    TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             ALL_OBJECTS, 0, 1, 1, ray, visible);
    return visible;
}

public struct LightSample {
    public float3 P;
    public float3 N;
    public float2 uv;
    public float  pdf;
};

public float lightPmf(MaterialGPU material, InstanceGPU instance) {
    return material.luminance * instance.lightArea * pc.emissivePowerInv;
}

public LightSample sampleEmissiveMesh(inout uint seed, InstanceGPU instance, MeshGPU mesh) {
    uint triCount = mesh.indexCount / 3u;
    bool useAlias = mesh.aliasAddress != 0 && mesh.area > 0.0;

    uint triIndex = min(uint(rand01(seed) * float(triCount)), triCount - 1u);
    if (useAlias) {
        AliasEntry* aliasTable = (AliasEntry*)mesh.aliasAddress;
        AliasEntry  entry      = aliasTable[triIndex];
        triIndex = (rand01(seed) < entry.prob) ? triIndex : entry.alias;
    }

    Vertex v0 = mesh.vertices[mesh.indices[triIndex * 3u + 0u]];
    Vertex v1 = mesh.vertices[mesh.indices[triIndex * 3u + 1u]];
    Vertex v2 = mesh.vertices[mesh.indices[triIndex * 3u + 2u]];

    float r1 = sqrt(rand01(seed));
    float r2 = rand01(seed);
    float bu = 1.0 - r1;
    float bv = r1 * (1.0 - r2);
    float bw = r1 * r2;

    float4x4 xform = instance.model;
    float3   lP    = mul(xform, float4(bu * v0.pos + bv * v1.pos + bw * v2.pos, 1.0)).xyz;

    float3 e1        = mul(xform, float4(v1.pos - v0.pos, 0.0)).xyz;
    float3 e2        = mul(xform, float4(v2.pos - v0.pos, 0.0)).xyz;
    float3 lN        = normalize(cross(e1, e2));
    float  worldArea = 0.5 * length(cross(e1, e2));
    float  objArea   = 0.5 * length(cross(v1.pos - v0.pos, v2.pos - v0.pos));

    float selectProb = useAlias ? (objArea / mesh.area) : (1.0 / float(triCount));

    LightSample ls;
    ls.P   = lP;
    ls.N   = lN;
    ls.uv  = bu * v0.texCoord + bv * v1.texCoord + bw * v2.texCoord;
    ls.pdf = (worldArea > PDF_EPS) ? (selectProb / worldArea) : 0.0;
    return ls;
}

public float emitterMisWeight(MeshGPU mesh, float objArea, float worldArea, float cosLight, float dist, float pmf, float prevBrdfPdf) {
    if (pc.nee == 0u || pc.emissiveCount == 0u || prevBrdfPdf < 0.0 ||
        worldArea <= PDF_EPS || cosLight <= PDF_EPS || pmf <= 0.0)
        return 1.0;
    uint  triCount   = mesh.indexCount / 3u;
    bool  useAlias   = mesh.aliasAddress != 0 && mesh.area > 0.0;
    float selectProb = useAlias ? (objArea / mesh.area) : (1.0 / float(triCount));
    float lightPdf   = pmf * (selectProb / worldArea) * dist * dist / cosLight;
    return powerHeuristic(prevBrdfPdf, lightPdf);
}

public void directLightContribution(inout PathPayload payload, BSDF bsdf, float3 wo, float3 P, float3 gN) {
    if (directLight.intensity <= 0.0) return;
    float3 L     = normalize(-directLight.dir);
    float  NdotL = max(dot(bsdf.ns, L), 0.0);
    if (NdotL <= 0.0) return;
    float vis = traceShadow(P, gN, L, T_MAX);
    payload.radiance += payload.throughput * bsdf.f(wo, L) * NdotL * directLight.intensity * vis;
}

public void sampledLightNEE(inout PathPayload payload, BSDF bsdf, float3 wo, float3 P, float3 gN) {
    if (pc.nee == 0u || pc.emissiveCount == 0u) return;

    uint        slot0         = min(uint(rand01(payload.seed) * float(pc.emissiveCount)), pc.emissiveCount - 1u);
    EmissiveGPU entry0        = emissiveLights[slot0];
    uint        slot          = (rand01(payload.seed) < entry0.aliasProb) ? slot0 : entry0.aliasIndex;
    uint        instanceIndex = emissiveLights[slot].instanceIndex;
    InstanceGPU lightInstance  = instances[instanceIndex];
    MeshGPU     lightMesh      = meshes[lightInstance.meshIndex];
    MaterialGPU lightMaterial  = materials[lightInstance.materialIndex];

    float pmf = lightPmf(lightMaterial, lightInstance);
    if (pmf <= 0.0) return;

    LightSample ls = sampleEmissiveMesh(payload.seed, lightInstance, lightMesh);
    if (ls.pdf <= 0.0) return;
    float3 emission = getBaseColor(lightMaterial, ls.uv) * lightMaterial.luminance;

    float3 toLight = ls.P - P;
    float  distSq  = dot(toLight, toLight);
    float  dist    = sqrt(distSq);
    float3 wi      = toLight / dist;

    float NdotWi  = max(dot(bsdf.ns, wi), 0.0);
    float LNdotWi = abs(dot(ls.N, wi));
    if (NdotWi <= 0.0 || LNdotWi <= PDF_EPS || distSq <= PDF_EPS) return;

    float  lightPdf = pmf * ls.pdf * distSq / LNdotWi;
    float  bsdfPdf  = bsdf.pdf(wo, wi);
    float  weight   = powerHeuristic(lightPdf, bsdfPdf);

    float vis = traceShadow(P, gN, wi, dist * (1.0 - SHADOW_EPS));
    payload.radiance += payload.throughput * bsdf.f(wo, wi) * NdotWi * emission * weight * vis / lightPdf;
}
