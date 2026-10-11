module lighting;

import common;
import types;
import scene;
import bsdf;
import fog;

public float traceShadow(float3 P, float3 N, float3 L, float tMax, inout uint seed) {
    float3 origin        = movedPoint(P, N);
    float  transmittance = fogTransmittance(origin, L, tMax);
    if (transmittance <= 0.0) return 0.0;

    ShadowPayload shadow;
    shadow.visible = 0.0;
    shadow.seed    = seed;
    RayDesc ray;
    ray.Origin    = origin;
    ray.TMin      = T_MIN;
    ray.Direction = L;
    ray.TMax      = tMax - T_MIN;
    TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
             ALL_OBJECTS, 0, 1, 1, ray, shadow);
    seed = shadow.seed;
    return shadow.visible * transmittance;
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
    LightSample ls;
    ls.P   = float3(0.0);
    ls.N   = float3(0.0, 1.0, 0.0);
    ls.uv  = float2(0.0);
    ls.pdf = 0.0;

    uint triCount = mesh.indexCount / 3u;
    if (triCount == 0u) return ls;
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
    float vis = traceShadow(P, gN, L, T_MAX, payload.seed);
    payload.radiance += payload.throughput * bsdf.f(wo, L) * NdotL * directLight.intensity * vis;
}

public struct EmitterPick {
    public InstanceGPU instance;
    public MeshGPU     mesh;
    public MaterialGPU material;
    public float       pmf;
};

public bool pickEmitter(inout uint seed, out EmitterPick pick) {
    uint        slot0  = min(uint(rand01(seed) * float(pc.emissiveCount)), pc.emissiveCount - 1u);
    EmissiveGPU entry0 = emissiveLights[slot0];
    uint        slot   = (rand01(seed) < entry0.aliasProb) ? slot0 : entry0.aliasIndex;

    pick.instance = instances[emissiveLights[slot].instanceIndex];
    pick.mesh     = meshes[pick.instance.meshIndex];
    pick.material = materials[pick.instance.materialIndex];
    pick.pmf      = lightPmf(pick.material, pick.instance);
    return pick.pmf > 0.0;
}

public void sampledLightNEE(inout PathPayload payload, BSDF bsdf, float3 wo, float3 P, float3 gN) {
    if (pc.nee == 0u || pc.emissiveCount == 0u) return;

    EmitterPick pick;
    if (!pickEmitter(payload.seed, pick)) return;
    float pmf = pick.pmf;

    LightSample ls = sampleEmissiveMesh(payload.seed, pick.instance, pick.mesh);
    if (ls.pdf <= 0.0) return;
    float3 emission = getBaseColor(pick.material, ls.uv) * pick.material.luminance;

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

    float vis = traceShadow(P, gN, wi, dist * (1.0 - SHADOW_EPS), payload.seed);
    payload.radiance += payload.throughput * bsdf.f(wo, wi) * NdotWi * emission * weight * vis / lightPdf;
}

public void sampledEnvNEE(inout PathPayload payload, BSDF bsdf, float3 wo, float3 P, float3 gN) {
    EnvSample es = sampleEnvLight(payload.seed);
    if (es.pdf <= PDF_EPS) return;

    float3 wi    = es.dir;
    float  NdotWi = max(dot(bsdf.ns, wi), 0.0);
    if (NdotWi <= 0.0) return;

    float bsdfPdf = bsdf.pdf(wo, wi);
    float weight  = powerHeuristic(es.pdf, bsdfPdf);

    float vis = traceShadow(P, gN, wi, T_MAX, payload.seed);
    payload.radiance += payload.throughput * bsdf.f(wo, wi) * NdotWi * es.radiance * weight * vis / es.pdf;
}

void fogDirectLight(inout PathPayload payload, float3 P, float3 dir, float g) {
    if (directLight.intensity <= 0.0) return;
    float3 L   = normalize(-directLight.dir);
    float  vis = traceShadow(P, float3(0.0), L, T_MAX, payload.seed);
    if (vis <= 0.0) return;
    payload.radiance += payload.throughput * hgPhase(dot(dir, L), g) * directLight.intensity * vis;
}

void fogLightNEE(inout PathPayload payload, float3 P, float3 dir, float g) {
    if (pc.nee == 0u || pc.emissiveCount == 0u) return;

    EmitterPick pick;
    if (!pickEmitter(payload.seed, pick)) return;

    LightSample ls = sampleEmissiveMesh(payload.seed, pick.instance, pick.mesh);
    if (ls.pdf <= 0.0) return;
    float3 emission = getBaseColor(pick.material, ls.uv) * pick.material.luminance;

    float3 toLight = ls.P - P;
    float  distSq  = dot(toLight, toLight);
    if (distSq <= PDF_EPS) return;
    float  dist    = sqrt(distSq);
    float3 wi      = toLight / dist;

    float LNdotWi = abs(dot(ls.N, wi));
    if (LNdotWi <= PDF_EPS) return;

    float lightPdf = pick.pmf * ls.pdf * distSq / LNdotWi;
    float phase    = hgPhase(dot(dir, wi), g);
    float weight   = powerHeuristic(lightPdf, phase);

    float vis = traceShadow(P, float3(0.0), wi, dist * (1.0 - SHADOW_EPS), payload.seed);
    payload.radiance += payload.throughput * phase * emission * weight * vis / lightPdf;
}

void fogEnvNEE(inout PathPayload payload, float3 P, float3 dir, float g) {
    EnvSample es = sampleEnvLight(payload.seed);
    if (es.pdf <= PDF_EPS) return;

    float phase  = hgPhase(dot(dir, es.dir), g);
    float weight = powerHeuristic(es.pdf, phase);

    float vis = traceShadow(P, float3(0.0), es.dir, T_MAX, payload.seed);
    payload.radiance += payload.throughput * phase * es.radiance * weight * vis / es.pdf;
}

public bool fogScatter(inout PathPayload payload, float tMax) {
    float3 origin = WorldRayOrigin();
    float3 dir    = WorldRayDirection();
    float  t;
    if (!fogSampleDistance(origin, dir, tMax, payload.seed, t)) return false;

    float3 P = origin + dir * t;
    float  g = fogData.anisotropy;
    payload.throughput *= fogData.color;

    fogDirectLight(payload, P, dir, g);
    fogLightNEE(payload, P, dir, g);
    fogEnvNEE(payload, P, dir, g);

    float pdf;
    payload.direction   = hgSample(dir, g, payload.seed, pdf);
    payload.origin      = P;
    payload.prevBrdfPdf = pdf;
    payload.segmentDist = 0.0;
    payload.done        = false;
    payload.passthrough = false;
    return true;
}
