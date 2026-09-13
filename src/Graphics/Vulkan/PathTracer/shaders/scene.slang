module scene;

import common;
import types;
import geometry;

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
[[vk::binding(5)]] public StructuredBuffer<MeshGPU>       meshes;
[[vk::binding(6)]] public StructuredBuffer<InstanceGPU>   instances;
[[vk::binding(7)]] public StructuredBuffer<EmissiveGPU>   emissiveLights;
[[vk::binding(8)]] public StructuredBuffer<MaterialGPU>   materials;
[[vk::binding(9)]] public Sampler2D                       textures[];

public float3 sampleEnvironment(float3 dir) {
    if (pc.skyboxIndex == UINT_MAX) return pc.skyColor;
    float2 uv = float2(atan2(dir.z, dir.x) * 0.15915494309 + 0.5,
                       acos(clamp(-dir.y, -1.0, 1.0)) * M_1_PI);
    float3 c = textures[pc.skyboxIndex].SampleLevel(uv, 0.0).rgb;
    return clamp(c, float3(0.0), float3(65504.0));
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

public void applyNormalMap(inout Hit hit, MaterialGPU material) {
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

public void primaryRay(uint2 pixel, uint2 imageSize, inout uint seed, out float3 origin, out float3 dir) {
    float2 jitter = float2(rand01(seed), rand01(seed)) - 0.5;
    float2 uv     = (float2(pixel) + jitter) / float2(imageSize);
    float4 target = mul(camera.invProj, float4(uv * 2.0 - 1.0, 1.0, 1.0));
    float3 dirVS  = normalize(target.xyz);
    float3 oriVS  = float3(0.0);

    if (pc.aperture > 0.0) {
        float  t       = pc.focusDistance / max(-dirVS.z, 1e-4);
        float3 focusVS = dirVS * t;
        float  r       = sqrt(rand01(seed)) * pc.aperture;
        float  phi     = 2.0 * M_PI * rand01(seed);
        oriVS = float3(r * cos(phi), r * sin(phi), 0.0);
        dirVS = normalize(focusVS - oriVS);
    }

    origin = mul(camera.invView, float4(oriVS, 1.0)).xyz;
    dir    = normalize(mul(camera.invView, float4(dirVS, 0.0)).xyz);
}
