module scene;                                         // Slang module: bindings + material/camera helpers + light/env sampling

import common;                                        // pc, constants, MIS helpers
import types;                                         // MaterialGPU/MeshGPU/InstanceGPU
import bsdf;                                          // Surface, evalBSDF (used by NEE)
import lighting;                                      // LightSample, emissive-mesh sampling
import geometry;                                      // HitGeometry (applyNormalMap)

public struct CameraData {                            // camera UBO layout
    public float4x4 invView;                          // view->world
    public float4x4 invProj;                          // clip->view
};                                                    // end CameraData

public struct DirectLightData {                       // directional-light UBO layout
    public float3 dir;                                // travel direction of the light
    public float  intensity;                          // intensity (0 disables it)
};                                                    // end DirectLightData

[[vk::binding(0)]] public RaytracingAccelerationStructure tlas; // binding 0: TLAS
[[vk::image_format("rgba16f")]] [[vk::binding(1)]] public RWTexture2D<float4> outputImage; // binding 1: HDR accumulation image
[[vk::image_format("r32ui")]]   [[vk::binding(2)]] public RWTexture2D<uint>   outputInstanceIdImage; // binding 2: instance-id image
[[vk::binding(3)]] public ConstantBuffer<CameraData>      camera;      // binding 3: camera UBO
[[vk::binding(4)]] public ConstantBuffer<DirectLightData> directLight; // binding 4: directional-light UBO
[[vk::binding(5)]] public StructuredBuffer<MeshGPU>      meshes;       // binding 5: mesh records
[[vk::binding(6)]] public StructuredBuffer<InstanceGPU>   instances;    // binding 6: instance records
[[vk::binding(7)]] public StructuredBuffer<uint>          emissiveIndices; // binding 7: emissive instance indices
[[vk::binding(8)]] public StructuredBuffer<MaterialGPU>  materials;    // binding 8: material records
[[vk::binding(9)]] public Sampler2D                       textures[];   // binding 9: bindless texture array

public float3 sampleEnvironment(float3 dir) {         // background/env radiance for a direction
    if (pc.skyboxIndex == UINT_MAX) return pc.skyColor; // flat sky if no env map
    float2 uv = float2(atan2(dir.z, dir.x) * 0.15915494309 + 0.5, // equirect U (azimuth / 2pi)
                       acos(clamp(-dir.y, -1.0, 1.0)) * 0.31830988618); // equirect V (polar / pi)
    float3 c = textures[pc.skyboxIndex].SampleLevel(uv, 0.0).rgb; // sample env map
    return clamp(c, float3(0.0), float3(65504.0));    // clamp to max half-float
}                                                     // end sampleEnvironment

public struct EnvSample {                             // a direction sampled from the environment map
    public float3 dir;                                // sampled world-space direction
    public float3 radiance;                           // env radiance along it
    public float  pdf;                                // solid-angle pdf
};                                                    // end EnvSample

bool envSamplingEnabled() {                           // is environment importance sampling usable this frame?
    return pc.envNee != 0u && pc.skyboxIndex != UINT_MAX && pc.envIntegral > 0.0; // need env NEE on, a map, and a valid integral
}                                                     // end envSamplingEnabled

int2 envTexSize() {                                   // dimensions of the bound environment map
    uint w, h;                                        // width/height out params
    textures[pc.skyboxIndex].GetDimensions(w, h);     // query texture size
    return int2(int(w), int(h));                      // return as ints
}                                                     // end envTexSize

int sampleCdfInterval(float* cdf, int base, int n, float u) { // binary search a CDF slice for the interval containing u
    int lo = 0;                                       // low bound
    int hi = n;                                       // high bound
    while (lo + 1 < hi) {                             // bisection loop
        int mid = (lo + hi) >> 1;                     // midpoint
        if (cdf[base + mid] <= u) lo = mid;           // u is to the right
        else                      hi = mid;           // u is to the left
    }                                                 // end bisection
    return lo;                                        // index of the chosen interval
}                                                     // end sampleCdfInterval

public float envPdf(float3 dir) {                     // pdf of sampling `dir` under env importance sampling (for MIS)
    if (!envSamplingEnabled()) return 0.0;            // disabled -> zero pdf
    float theta = acos(clamp(-dir.y, -1.0, 1.0));     // polar angle
    float sinT  = sin(theta);                         // sin(theta) for the Jacobian
    if (sinT <= 0.0) return 0.0;                      // degenerate at the poles
    float u = atan2(dir.z, dir.x) * 0.15915494309 + 0.5; // equirect U
    float v = theta * 0.31830988618;                  // equirect V
    int2  sz = envTexSize();                          // map size
    int   nu = sz.x;                                  // columns
    int   nv = sz.y;                                  // rows
    int   iu = clamp(int(u * float(nu)), 0, nu - 1);  // column index
    int   iv = clamp(int(v * float(nv)), 0, nv - 1);  // row index
    float* condFunc = (float*)pc.envCondFuncAddr;     // pointer to per-texel luminance function
    float mapPdf = condFunc[iv * nu + iu] / pc.envIntegral; // pixel-space pdf (normalized by the integral)
    return mapPdf / (2.0 * M_PI * M_PI * sinT);       // convert pixel-space pdf to solid-angle pdf
}                                                     // end envPdf

public EnvSample sampleEnvLight(inout uint seed) {    // importance-sample a direction from the environment map
    EnvSample es;                                     // output
    es.dir      = float3(0.0, 1.0, 0.0);              // default direction
    es.radiance = float3(0.0);                        // default radiance
    es.pdf      = 0.0;                                 // default pdf
    if (!envSamplingEnabled()) return es;             // disabled -> empty sample

    int2 sz = envTexSize();                           // map size
    int  nu = sz.x;                                   // columns
    int  nv = sz.y;                                   // rows
    float* marginalCdf = (float*)pc.envMarginalCdfAddr; // CDF over rows
    float* condCdf     = (float*)pc.envCondCdfAddr;   // per-row CDFs over columns
    float* condFunc    = (float*)pc.envCondFuncAddr;  // per-texel luminance function

    float u0 = rand01(seed);                          // random for the column
    float u1 = rand01(seed);                          // random for the row

    int   iv   = sampleCdfInterval(marginalCdf, 0, nv, u1); // pick a row from the marginal CDF
    float dv   = u1 - marginalCdf[iv];                // fractional position within the row interval
    float mden = marginalCdf[iv + 1] - marginalCdf[iv]; // interval width
    if (mden > 0.0) dv /= mden;                       // normalize to [0,1)
    float v = (float(iv) + dv) / float(nv);           // continuous V coordinate

    int   rowBase = iv * (nu + 1);                    // offset into the conditional CDF for this row
    int   iu   = sampleCdfInterval(condCdf, rowBase, nu, u0); // pick a column from the conditional CDF
    float du   = u0 - condCdf[rowBase + iu];          // fractional position within the column interval
    float cden = condCdf[rowBase + iu + 1] - condCdf[rowBase + iu]; // interval width
    if (cden > 0.0) du /= cden;                       // normalize to [0,1)

    float mapPdf = condFunc[iv * nu + iu] / pc.envIntegral; // pixel-space pdf of the chosen texel

    float theta = v * M_PI;                           // polar angle from V
    float phi   = ((float(iu) + du) / float(nu) - 0.5) * 2.0 * M_PI; // azimuth from U
    float sinT  = sin(theta);                         // sin(theta)
    if (sinT <= 0.0 || mapPdf <= 0.0) return es;      // degenerate -> empty sample

    es.dir      = float3(sinT * cos(phi), -cos(theta), sinT * sin(phi)); // direction from spherical coords
    es.pdf      = mapPdf / (2.0 * M_PI * M_PI * sinT); // solid-angle pdf
    es.radiance = sampleEnvironment(es.dir);          // radiance along the sampled direction
    return es;                                         // env light sample
}                                                     // end sampleEnvLight

public float3 getBaseColor(MaterialGPU material, float2 uv) { // albedo * optional base-color texture
    float3 baseColor = material.baseColor;            // base color factor
    if (material.baseColorTexIndex != UINT_MAX)       // texture assigned?
        baseColor *= textures[NonUniformResourceIndex(material.baseColorTexIndex)].SampleLevel(uv, 0.0).rgb; // modulate
    return baseColor;                                 // final albedo
}                                                     // end getBaseColor

public void applyMetalRoughness(MaterialGPU material, float2 uv, inout float metalness, inout float roughness) { // fold metal/rough texture
    if (material.metalRoughnessTexIndex == UINT_MAX) return; // no texture
    float3 mr = textures[NonUniformResourceIndex(material.metalRoughnessTexIndex)].SampleLevel(uv, 0.0).rgb; // sample
    roughness *= mr.g;                                // green = roughness
    metalness *= mr.b;                                // blue = metalness
}                                                     // end applyMetalRoughness

public void applyClearcoat(MaterialGPU material, float2 uv, inout float clearcoat, inout float clearcoatRoughness) { // fold clearcoat textures
    if (material.clearcoatTexIndex != UINT_MAX)       // clearcoat strength texture?
        clearcoat *= textures[NonUniformResourceIndex(material.clearcoatTexIndex)].SampleLevel(uv, 0.0).r; // red
    if (material.clearcoatRoughnessTexIndex != UINT_MAX) // clearcoat roughness texture?
        clearcoatRoughness *= textures[NonUniformResourceIndex(material.clearcoatRoughnessTexIndex)].SampleLevel(uv, 0.0).g; // green
}                                                     // end applyClearcoat

public void applyNormalMap(inout HitGeometry hit, MaterialGPU material) { // perturb shading normal with a normal map
    if (material.normalTexIndex == UINT_MAX) return;  // no normal map
    if (dot(hit.tangent.xyz, hit.tangent.xyz) < 1e-8) return; // need a valid tangent
    float2 rg = textures[NonUniformResourceIndex(material.normalTexIndex)].SampleLevel(hit.uv, 0.0).rg * 2.0 - 1.0; // decode XY
    rg.y = -rg.y;                                     // flip green (convention)
    rg *= material.normalScale;                       // scale bump strength
    float3 tn = float3(rg, sqrt(saturate(1.0 - dot(rg, rg)))); // reconstruct Z
    float3 N = hit.N;                                 // current normal
    float3 T = normalize(hit.tangent.xyz - dot(hit.tangent.xyz, N) * N); // orthonormalize tangent
    float3 B = cross(N, T) * hit.tangent.w;           // bitangent with handedness
    hit.N = normalize(tn.x * T + tn.y * B + tn.z * N); // world-space perturbed normal
}                                                     // end applyNormalMap

public void primaryRay(uint2 pixel, uint2 size, inout uint seed, out float3 origin, out float3 dir) { // jittered camera ray with DoF
    float2 jitter = float2(rand01(seed), rand01(seed)) - 0.5; // sub-pixel jitter
    float2 uv     = (float2(pixel) + jitter) / float2(size); // normalized screen coordinate
    float4 target = mul(camera.invProj, float4(uv * 2.0 - 1.0, 1.0, 1.0)); // unproject to view space
    float3 dirVS  = normalize(target.xyz);            // view-space direction
    float3 oriVS  = float3(0.0);                      // pinhole origin at the eye

    if (pc.aperture > 0.0) {                          // depth of field?
        float  t       = pc.focusDistance / max(-dirVS.z, 1e-4); // distance to focal plane
        float3 focusVS = dirVS * t;                   // focal point in view space
        float  r       = sqrt(rand01(seed)) * pc.aperture; // lens-disk radius
        float  phi     = 2.0 * M_PI * rand01(seed);   // lens-disk angle
        oriVS = float3(r * cos(phi), r * sin(phi), 0.0); // jittered lens origin
        dirVS = normalize(focusVS - oriVS);           // direction through the focal point
    }                                                 // end DoF

    origin = mul(camera.invView, float4(oriVS, 1.0)).xyz; // origin to world space
    dir    = normalize(mul(camera.invView, float4(dirVS, 0.0)).xyz); // direction to world space
}                                                     // end primaryRay

public float traceShadow(float3 P, float3 N, float3 L, float tMax) { // visibility test from P toward direction L
    float visible = 0.0;                              // default occluded (set to 1 by the shadow miss shader)
    RayDesc ray;                                      // shadow ray
    ray.Origin    = movedPoint(P, N);                 // offset origin off the surface
    ray.TMin      = T_MIN;                            // start distance
    ray.Direction = L;                                // toward the light
    ray.TMax      = tMax - T_MIN;                     // stop just short of the light
    TraceRay(tlas,                                    // trace the shadow ray
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, // any-hit/early-out, skip closest-hit
        ALL_OBJECTS, 0, 1, 1, ray, visible);          // hit offset 0, shadow miss index 1
    return visible;                                   // 1 if unoccluded, 0 otherwise
}                                                     // end traceShadow

public void directLightContribution(inout PathPayload payload, float3 P, float3 gN, Surface surf) { // NEE for the directional light
    if (directLight.intensity <= 0.0) return;         // no directional light
    float3 L     = normalize(-directLight.dir);       // direction toward the light
    float  NdotL = max(dot(surf.N, L), 0.0);          // cosine at the surface
    if (NdotL <= 0.0) return;                          // light below the horizon
    float3 brdf;                                       // BRDF value out param
    float  pdf;                                        // BRDF pdf out param (unused here)
    evalBSDF(surf, L, brdf, pdf);                      // evaluate BRDF toward the light
    float  vis    = traceShadow(P, gN, L, T_MAX);      // visibility toward the light
    payload.radiance += payload.throughput * brdf * NdotL * directLight.intensity * vis; // add direct contribution
}                                                     // end directLightContribution

public struct LightCandidate {                        // a tentative NEE contribution before the shadow test
    public float3 wi;                                 // direction to the light
    public float3 value;                              // throughput-free contribution (brdf*Le*cos*mis/pdf)
    public float  tMax;                               // shadow-ray max distance
    public bool   valid;                              // whether the candidate carries any energy
};                                                    // end LightCandidate

LightCandidate emitterCandidate(inout uint seed, float3 P, Surface surf) { // build a candidate by sampling an emissive mesh
    LightCandidate c;                                 // output
    c.wi    = float3(0.0, 1.0, 0.0);                  // default direction
    c.value = float3(0.0);                            // default value
    c.tMax  = 0.0;                                     // default distance
    c.valid = false;                                   // default invalid

    uint lightIdx      = min(uint(rand01(seed) * float(pc.emissiveCount)), pc.emissiveCount - 1u); // pick an emissive instance
    uint instanceIndex = emissiveIndices[lightIdx];   // map to a global instance index
    InstanceGPU  lightInstance = instances[instanceIndex]; // instance record
    MeshGPU     lightMesh     = meshes[lightInstance.meshIndex]; // mesh record
    MaterialGPU lightMaterial = materials[lightInstance.materialIndex]; // material record
    LightSample  ls            = sampleEmissiveMesh(seed, lightInstance, lightMesh); // sample a point on the light
    ls.emission = getBaseColor(lightMaterial, ls.uv) * lightMaterial.luminance; // emitted radiance at the sample

    float3 toLight = ls.P - P;                        // vector to the light sample
    float  distSq  = dot(toLight, toLight);           // squared distance
    float  dist    = sqrt(distSq);                    // distance
    float3 wi      = toLight / dist;                  // unit direction to the light

    float NdotWi  = max(dot(surf.N, wi), 0.0);        // cosine at the shading point
    float LNdotWi = max(dot(ls.N, -wi), 0.0);         // cosine at the light surface
    if (NdotWi <= 0.0 || LNdotWi <= PDF_EPS || ls.pdf <= 0.0 || distSq <= PDF_EPS) return c; // reject degenerate geometry

    float3 brdf;                                       // BRDF value
    float  pdfBrdf;                                    // BRDF pdf (for MIS)
    evalBSDF(surf, wi, brdf, pdfBrdf);                 // evaluate BRDF toward the light
    float pdfLight = ls.pdf * distSq / LNdotWi / float(pc.emissiveCount); // light pdf in solid angle, including light selection
    float weight   = powerHeuristic(pdfLight, pdfBrdf); // MIS weight favoring light sampling

    c.wi    = wi;                                      // store direction
    c.value = brdf * ls.emission * NdotWi * weight / pdfLight; // MIS-weighted contribution (throughput applied later)
    c.tMax  = dist * (1.0 - SHADOW_EPS);               // shadow ray stops just before the light
    c.valid = true;                                    // candidate carries energy
    return c;                                          // emitter candidate
}                                                     // end emitterCandidate

LightCandidate envCandidate(inout uint seed, Surface surf) { // build a candidate by sampling the environment
    LightCandidate c;                                 // output
    c.wi    = float3(0.0, 1.0, 0.0);                  // default direction
    c.value = float3(0.0);                            // default value
    c.tMax  = 0.0;                                     // default distance
    c.valid = false;                                   // default invalid

    EnvSample es = sampleEnvLight(seed);              // importance-sample the env map
    if (es.pdf <= PDF_EPS) return c;                   // reject invalid sample
    float NdotL = max(dot(surf.N, es.dir), 0.0);       // cosine at the surface
    if (NdotL <= 0.0) return c;                         // below the horizon

    float3 brdf;                                       // BRDF value
    float  pdfBrdf;                                    // BRDF pdf
    evalBSDF(surf, es.dir, brdf, pdfBrdf);             // evaluate BRDF toward the env direction
    float weight = powerHeuristic(es.pdf, pdfBrdf);    // MIS weight favoring env sampling

    c.wi    = es.dir;                                  // store direction
    c.value = brdf * es.radiance * NdotL * weight / es.pdf; // MIS-weighted contribution
    c.tMax  = T_MAX;                                    // env is at infinity
    c.valid = true;                                     // candidate carries energy
    return c;                                           // env candidate
}                                                     // end envCandidate

public void sampledLightNEE(inout PathPayload payload, float3 P, float3 gN, Surface surf) { // NEE over emissive meshes and/or the environment
    bool meshActive = pc.nee != 0u && pc.emissiveCount != 0u; // mesh lights available?
    bool envActive  = envSamplingEnabled();           // env sampling available?
    if (!meshActive && !envActive) return;            // nothing to sample

    LightCandidate chosen;                            // the candidate we will shadow-test
    float selectProb = 1.0;                           // probability of choosing it (for unbiased weighting)

    if (meshActive && envActive) {                    // both sources available -> stochastically pick one
        LightCandidate cM = emitterCandidate(payload.seed, P, surf); // mesh candidate
        LightCandidate cE = envCandidate(payload.seed, surf); // env candidate
        float eM = cM.valid ? luminance(cM.value) : 0.0; // mesh candidate energy
        float eE = cE.valid ? luminance(cE.value) : 0.0; // env candidate energy
        if (eM <= 0.0 && eE <= 0.0) return;           // neither carries energy

        float pEnv;                                   // probability of choosing the env candidate
        if      (eM <= 0.0) pEnv = 1.0;               // only env has energy
        else if (eE <= 0.0) pEnv = 0.0;               // only mesh has energy
        else                pEnv = clamp(eE / (eE + eM), 0.05, 0.95); // balance by relative energy (clamped)

        if (rand01(payload.seed) < pEnv) { chosen = cE; selectProb = pEnv; } // pick env
        else                             { chosen = cM; selectProb = 1.0 - pEnv; } // pick mesh
    } else if (meshActive) {                          // only mesh lights
        chosen = emitterCandidate(payload.seed, P, surf); // mesh candidate
    } else {                                          // only environment
        chosen = envCandidate(payload.seed, surf);    // env candidate
    }                                                 // end source selection

    if (!chosen.valid) return;                        // nothing to add

    float  vis     = traceShadow(P, gN, chosen.wi, chosen.tMax); // visibility toward the chosen light
    float3 contrib = payload.throughput * chosen.value * vis / selectProb; // final contribution, undoing selection prob
    float  lum     = luminance(contrib);              // luminance for clamping
    if (lum > NEE_CLAMP) contrib *= NEE_CLAMP / lum;  // clamp fireflies
    payload.radiance += contrib;                      // accumulate the direct-lighting contribution
}                                                     // end sampledLightNEE
