module scene;

import common;
import types;
import bsdf;
import lighting;
import geometry;

public struct PushConstants {
    public uint frame;
    public uint spp;
    public uint minDepth;
    public uint maxDepth;
    public uint displayMode;
    public uint nee;
    public uint emissiveCount;
    public uint skyboxIndex;
    public float    envIntegral;
    public uint     envNee;
    public uint64_t envMarginalCdfAddr;
    public uint64_t envCondCdfAddr;
    public uint64_t envCondFuncAddr;
};
[[vk::push_constant]] public PushConstants pc;

public struct CameraData {
    public float4x4 invView;
    public float4x4 invProj;
};

public struct DirectLightData {
    public float3 dir;
    public float  intensity;
};

[[vk::binding(0)]] public RaytracingAccelerationStructure tlas;
[[vk::image_format("rgba16f")]] [[vk::binding(1)]] public RWTexture2D<float4> outputImage;
[[vk::image_format("r32ui")]]   [[vk::binding(2)]] public RWTexture2D<uint>   outputInstanceIdImage;
[[vk::binding(3)]] public ConstantBuffer<CameraData>      camera;
[[vk::binding(4)]] public ConstantBuffer<DirectLightData> directLight;
[[vk::binding(5)]] public StructuredBuffer<MeshGPU>      meshes;
[[vk::binding(6)]] public StructuredBuffer<InstanceGPU>   instances;
[[vk::binding(7)]] public StructuredBuffer<uint>          emissiveIndices;
[[vk::binding(8)]] public StructuredBuffer<MaterialGPU>  materials;
[[vk::binding(9)]] public Sampler2D                       textures[];

public float3 sampleEnvironment(float3 dir) {
    if (pc.skyboxIndex == UINT_MAX) return SKY_COLOR;
    float2 uv = float2(atan2(dir.z, dir.x) * 0.15915494309 + 0.5,
                       acos(clamp(-dir.y, -1.0, 1.0)) * 0.31830988618);
    float3 c = textures[pc.skyboxIndex].SampleLevel(uv, 0.0).rgb;
    return clamp(c, float3(0.0), float3(65504.0));
}

public struct EnvSample {
    public float3 dir;
    public float3 radiance;
    public float  pdf;
};

bool envSamplingEnabled() {
    return pc.envNee != 0u && pc.skyboxIndex != UINT_MAX && pc.envIntegral > 0.0;
}

int2 envTexSize() {
    uint w, h;
    textures[pc.skyboxIndex].GetDimensions(w, h);
    return int2(int(w), int(h));
}

int sampleCdfInterval(float* cdf, int base, int n, float u) {
    int lo = 0;
    int hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[base + mid] <= u) lo = mid;
        else                      hi = mid;
    }
    return lo;
}

public float envPdf(float3 dir) {
    if (!envSamplingEnabled()) return 0.0;
    float theta = acos(clamp(-dir.y, -1.0, 1.0));
    float sinT  = sin(theta);
    if (sinT <= 0.0) return 0.0;
    float u = atan2(dir.z, dir.x) * 0.15915494309 + 0.5;
    float v = theta * 0.31830988618;
    int2  sz = envTexSize();
    int   nu = sz.x;
    int   nv = sz.y;
    int   iu = clamp(int(u * float(nu)), 0, nu - 1);
    int   iv = clamp(int(v * float(nv)), 0, nv - 1);
    float* condFunc = (float*)pc.envCondFuncAddr;
    float mapPdf = condFunc[iv * nu + iu] / pc.envIntegral;
    return mapPdf / (2.0 * M_PI * M_PI * sinT);
}

public EnvSample sampleEnvLight(inout uint seed) {
    EnvSample es;
    es.dir      = float3(0.0, 1.0, 0.0);
    es.radiance = float3(0.0);
    es.pdf      = 0.0;
    if (!envSamplingEnabled()) return es;

    int2 sz = envTexSize();
    int  nu = sz.x;
    int  nv = sz.y;
    float* marginalCdf = (float*)pc.envMarginalCdfAddr;
    float* condCdf     = (float*)pc.envCondCdfAddr;
    float* condFunc    = (float*)pc.envCondFuncAddr;

    float u0 = rand01(seed);
    float u1 = rand01(seed);

    int   iv   = sampleCdfInterval(marginalCdf, 0, nv, u1);
    float dv   = u1 - marginalCdf[iv];
    float mden = marginalCdf[iv + 1] - marginalCdf[iv];
    if (mden > 0.0) dv /= mden;
    float v = (float(iv) + dv) / float(nv);

    int   rowBase = iv * (nu + 1);
    int   iu   = sampleCdfInterval(condCdf, rowBase, nu, u0);
    float du   = u0 - condCdf[rowBase + iu];
    float cden = condCdf[rowBase + iu + 1] - condCdf[rowBase + iu];
    if (cden > 0.0) du /= cden;

    float mapPdf = condFunc[iv * nu + iu] / pc.envIntegral;

    float theta = v * M_PI;
    float phi   = ((float(iu) + du) / float(nu) - 0.5) * 2.0 * M_PI;
    float sinT  = sin(theta);
    if (sinT <= 0.0 || mapPdf <= 0.0) return es;

    es.dir      = float3(sinT * cos(phi), -cos(theta), sinT * sin(phi));
    es.pdf      = mapPdf / (2.0 * M_PI * M_PI * sinT);
    es.radiance = sampleEnvironment(es.dir);
    return es;
}

public float3 getBaseColor(MaterialGPU material, float2 uv) {
    float3 baseColor = material.baseColor;
    if (material.baseColorTexIndex != UINT_MAX)
        baseColor *= textures[NonUniformResourceIndex(material.baseColorTexIndex)].SampleLevel(uv, 0.0).rgb;
    return baseColor;
}

public void applyMetalRoughness(MaterialGPU material, float2 uv, inout float metalness, inout float roughness) {
    if (material.metalRoughnessTexIndex == UINT_MAX) return;
    float3 mr = textures[NonUniformResourceIndex(material.metalRoughnessTexIndex)].SampleLevel(uv, 0.0).rgb;
    roughness *= mr.g;
    metalness *= mr.b;
}

public void applyClearcoat(MaterialGPU material, float2 uv, inout float clearcoat, inout float clearcoatRoughness) {
    if (material.clearcoatTexIndex != UINT_MAX)
        clearcoat *= textures[NonUniformResourceIndex(material.clearcoatTexIndex)].SampleLevel(uv, 0.0).r;
    if (material.clearcoatRoughnessTexIndex != UINT_MAX)
        clearcoatRoughness *= textures[NonUniformResourceIndex(material.clearcoatRoughnessTexIndex)].SampleLevel(uv, 0.0).g;
}

public void applyNormalMap(inout HitGeometry hit, MaterialGPU material) {
    if (material.normalTexIndex == UINT_MAX) return;
    if (dot(hit.tangent.xyz, hit.tangent.xyz) < 1e-8) return;
    float2 rg = textures[NonUniformResourceIndex(material.normalTexIndex)].SampleLevel(hit.uv, 0.0).rg * 2.0 - 1.0;
    rg.y = -rg.y;
    rg *= material.normalScale;
    float3 tn = float3(rg, sqrt(saturate(1.0 - dot(rg, rg))));
    float3 N = hit.N;
    float3 T = normalize(hit.tangent.xyz - dot(hit.tangent.xyz, N) * N);
    float3 B = cross(N, T) * hit.tangent.w;
    hit.N = normalize(tn.x * T + tn.y * B + tn.z * N);
}

public void primaryRay(uint2 pixel, uint2 size, inout uint seed, out float3 origin, out float3 dir) {
    float2 jitter = float2(rand01(seed), rand01(seed)) - 0.5;
    float2 uv     = (float2(pixel) + jitter) / float2(size);
    float4 target = mul(camera.invProj, float4(uv * 2.0 - 1.0, 1.0, 1.0));
    dir    = normalize(mul(camera.invView, float4(normalize(target.xyz), 0.0)).xyz);
    origin = mul(camera.invView, float4(0.0, 0.0, 0.0, 1.0)).xyz;
}

public float traceShadow(float3 P, float3 N, float3 L, float tMax) {
    float visible = 0.0;
    RayDesc ray;
    ray.Origin    = movedPoint(P, N);
    ray.TMin      = T_MIN;
    ray.Direction = L;
    ray.TMax      = tMax - T_MIN;
    TraceRay(tlas,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        ALL_OBJECTS, 0, 1, 1, ray, visible);
    return visible;
}

public void directLightContribution(inout PathPayload payload, float3 P, float3 gN, Surface surf) {
    if (directLight.intensity <= 0.0) return;
    float3 L     = normalize(-directLight.dir);
    float  NdotL = max(dot(surf.N, L), 0.0);
    if (NdotL <= 0.0) return;
    float3 brdf;
    float  pdf;
    evalBSDF(surf, L, brdf, pdf);
    float  vis    = traceShadow(P, gN, L, T_MAX);
    payload.radiance += payload.throughput * brdf * NdotL * directLight.intensity * vis;
}

public struct LightCandidate {
    public float3 wi;
    public float3 value;
    public float  tMax;
    public bool   valid;
};

LightCandidate emitterCandidate(inout uint seed, float3 P, Surface surf) {
    LightCandidate c;
    c.wi    = float3(0.0, 1.0, 0.0);
    c.value = float3(0.0);
    c.tMax  = 0.0;
    c.valid = false;

    uint lightIdx      = min(uint(rand01(seed) * float(pc.emissiveCount)), pc.emissiveCount - 1u);
    uint instanceIndex = emissiveIndices[lightIdx];
    InstanceGPU  lightInstance = instances[instanceIndex];
    MeshGPU     lightMesh     = meshes[lightInstance.meshIndex];
    MaterialGPU lightMaterial = materials[lightInstance.materialIndex];
    LightSample  ls            = sampleEmissiveMesh(seed, lightInstance, lightMesh);
    ls.emission = getBaseColor(lightMaterial, ls.uv) * lightMaterial.luminance;

    float3 toLight = ls.P - P;
    float  distSq  = dot(toLight, toLight);
    float  dist    = sqrt(distSq);
    float3 wi      = toLight / dist;

    float NdotWi  = max(dot(surf.N, wi), 0.0);
    float LNdotWi = max(dot(ls.N, -wi), 0.0);
    if (NdotWi <= 0.0 || LNdotWi <= PDF_EPS || ls.pdf <= 0.0 || distSq <= PDF_EPS) return c;

    float3 brdf;
    float  pdfBrdf;
    evalBSDF(surf, wi, brdf, pdfBrdf);
    float pdfLight = ls.pdf * distSq / LNdotWi / float(pc.emissiveCount);
    float weight   = powerHeuristic(pdfLight, pdfBrdf);

    c.wi    = wi;
    c.value = brdf * ls.emission * NdotWi * weight / pdfLight;
    c.tMax  = dist * (1.0 - SHADOW_EPS);
    c.valid = true;
    return c;
}

LightCandidate envCandidate(inout uint seed, Surface surf) {
    LightCandidate c;
    c.wi    = float3(0.0, 1.0, 0.0);
    c.value = float3(0.0);
    c.tMax  = 0.0;
    c.valid = false;

    EnvSample es = sampleEnvLight(seed);
    if (es.pdf <= PDF_EPS) return c;
    float NdotL = max(dot(surf.N, es.dir), 0.0);
    if (NdotL <= 0.0) return c;

    float3 brdf;
    float  pdfBrdf;
    evalBSDF(surf, es.dir, brdf, pdfBrdf);
    float weight = powerHeuristic(es.pdf, pdfBrdf);

    c.wi    = es.dir;
    c.value = brdf * es.radiance * NdotL * weight / es.pdf;
    c.tMax  = T_MAX;
    c.valid = true;
    return c;
}

public void sampledLightNEE(inout PathPayload payload, float3 P, float3 gN, Surface surf) {
    bool meshActive = pc.nee != 0u && pc.emissiveCount != 0u;
    bool envActive  = envSamplingEnabled();
    if (!meshActive && !envActive) return;

    LightCandidate chosen;
    float selectProb = 1.0;

    if (meshActive && envActive) {
        LightCandidate cM = emitterCandidate(payload.seed, P, surf);
        LightCandidate cE = envCandidate(payload.seed, surf);
        float eM = cM.valid ? luminance(cM.value) : 0.0;
        float eE = cE.valid ? luminance(cE.value) : 0.0;
        if (eM <= 0.0 && eE <= 0.0) return;

        float pEnv;
        if      (eM <= 0.0) pEnv = 1.0;
        else if (eE <= 0.0) pEnv = 0.0;
        else                pEnv = clamp(eE / (eE + eM), 0.05, 0.95);

        if (rand01(payload.seed) < pEnv) { chosen = cE; selectProb = pEnv; }
        else                             { chosen = cM; selectProb = 1.0 - pEnv; }
    } else if (meshActive) {
        chosen = emitterCandidate(payload.seed, P, surf);
    } else {
        chosen = envCandidate(payload.seed, surf);
    }

    if (!chosen.valid) return;

    float  vis     = traceShadow(P, gN, chosen.wi, chosen.tMax);
    float3 contrib = payload.throughput * chosen.value * vis / selectProb;
    float  lum     = luminance(contrib);
    if (lum > NEE_CLAMP) contrib *= NEE_CLAMP / lum;
    payload.radiance += contrib;
}
