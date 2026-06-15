module scene;

import common;
import types;
import bsdf;
import lighting;

public struct PushConstants {
    public uint frame;
    public uint spp;
    public uint minDepth;
    public uint maxDepth;
    public uint displayMode;
    public uint nee;
    public uint emissiveCount;
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
[[vk::binding(5)]] public StructuredBuffer<MeshInfo>      meshes;
[[vk::binding(6)]] public StructuredBuffer<InstanceData>  instances;
[[vk::binding(7)]] public StructuredBuffer<uint>          emissiveIndices;
[[vk::binding(8)]] public StructuredBuffer<MaterialData>  materials;
[[vk::binding(9)]] public Sampler2D                       textures[];

public float3 getBaseColor(MaterialData material, float2 uv) {
    float3 baseColor = material.baseColor;
    if (material.baseColorTexIndex != UINT_MAX)
        baseColor *= textures[NonUniformResourceIndex(material.baseColorTexIndex)].SampleLevel(uv, 0.0).rgb;
    return baseColor;
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

public void directLightContribution(inout PathPayload payload, float3 P, float3 gN, float3 N, Scatter scatter) {
    if (directLight.intensity <= 0.0) return;
    float3 L     = normalize(-directLight.dir);
    float  NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return;
    float  vis    = traceShadow(P, gN, L, T_MAX);
    payload.radiance += payload.throughput * scatter.brdf * NdotL * directLight.intensity * vis;
}

public void nextEventEstimation(inout PathPayload payload, float3 P, float3 gN, float3 N, Scatter scatter) {
    if (pc.nee == 0u || pc.emissiveCount == 0u) return;

    uint lightIdx      = min(uint(rand01(payload.seed) * float(pc.emissiveCount)), pc.emissiveCount - 1u);
    uint instanceIndex = emissiveIndices[lightIdx];
    InstanceData lightInstance = instances[instanceIndex];
    MeshInfo     lightMesh     = meshes[lightInstance.meshIndex];
    MaterialData lightMaterial = materials[lightInstance.materialIndex];
    LightSample  ls            = sampleEmissiveMesh(payload.seed, lightInstance, lightMesh, lightMaterial);

    float3 toLight = ls.P - P;
    float  distSq  = dot(toLight, toLight);
    float  dist    = sqrt(distSq);
    float3 wi      = toLight / dist;

    float NdotWi  = max(dot(N, wi), 0.0);
    float LNdotWi = max(dot(ls.N, -wi), 0.0);

    if (NdotWi <= 0.0 || LNdotWi <= PDF_EPS || ls.pdf <= 0.0 || distSq <= PDF_EPS) return;

    float vis      = traceShadow(P, gN, wi, dist * (1.0 - SHADOW_EPS));
    float pdfLight = ls.pdf * distSq / LNdotWi / float(pc.emissiveCount);
    float pdfBrdf  = NdotWi * M_1_PI;
    float weight   = powerHeuristic(pdfLight, pdfBrdf);

    float3 contrib = payload.throughput * scatter.brdf * ls.emission * NdotWi * vis * weight / pdfLight;
    float  lum     = luminance(contrib);
    if (lum > NEE_CLAMP) contrib *= NEE_CLAMP / lum;
    payload.radiance += contrib;
}
