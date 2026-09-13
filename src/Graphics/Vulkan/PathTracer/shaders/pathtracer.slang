import common;
import types;
import geometry;
import scene;
import bsdf;
import lighting;

[shader("raygeneration")]
void rgenMain() {
    uint2 pixel     = DispatchRaysIndex().xy;
    uint2 imageSize = DispatchRaysDimensions().xy;
    uint  pixelId   = pixel.x + pixel.y * imageSize.x;

    uint   sampleCount = max(pc.spp, 1u);
    float3 radianceSum = float3(0.0);
    for (uint sample = 0; sample < sampleCount; ++sample) {
        uint seed = pcg(pixelId + sample * 0x9e3779b9u);
        float3 origin, dir;
        primaryRay(pixel, imageSize, seed, origin, dir);

        PathPayload payload;
        payload.throughput  = float3(1.0);
        payload.radiance    = float3(0.0);
        payload.origin      = origin;
        payload.direction   = dir;
        payload.instanceId  = UINT_MAX;
        payload.seed        = seed;
        payload.prevBrdfPdf = -1.0;
        payload.done        = false;

        for (uint depth = 0; depth < pc.maxDepth; ++depth) {
            RayDesc ray;
            ray.Origin    = payload.origin;
            ray.TMin      = 0.0;
            ray.Direction = payload.direction;
            ray.TMax      = T_MAX;
            TraceRay(tlas, RAY_FLAG_NONE, ALL_OBJECTS, 0, 0, 0, ray, payload);

            if (depth == 0 && sample == 0)
                outputInstanceIdImage[pixel] = payload.instanceId + 1;
            if (payload.done) break;

            if (depth >= pc.minDepth) {
                float prob = clamp(luminance(payload.throughput), 0.01, 1.0);
                if (rand01(payload.seed) > prob) break;
                payload.throughput /= prob;
            }
        }
        radianceSum += payload.radiance;
    }

    float3 radiance = radianceSum / float(sampleCount);
    if (any(isnan(radiance)) || any(isinf(radiance)))
        radiance = float3(0.0);

    if (pc.frame == 0) {
        outputImage[pixel] = float4(radiance, 1.0);
        return;
    }
    float3 prev = outputImage[pixel].rgb;
    outputImage[pixel] = float4((prev * float(pc.frame) + radiance) / float(pc.frame + 1), 1.0);
}

[shader("miss")]
void missMain(inout PathPayload payload) {
    payload.radiance += payload.throughput * sampleEnvironment(WorldRayDirection());
    payload.done = true;
}

[shader("miss")]
void shadowMissMain(inout float visible) {
    visible = 1.0;
}

float3 debugAOV(MaterialGPU material, Hit hit, float3 baseColor) {
    float metalness = material.metalness;
    float roughness = material.roughness;
    applyMetalRoughness(material, hit.uv, metalness, roughness);
    float clearcoat = material.clearcoat;
    float clearcoatRoughness = material.clearcoatRoughness;
    applyClearcoat(material, hit.uv, clearcoat, clearcoatRoughness);

    if (pc.displayMode == 2u) return hit.N * 0.5 + 0.5;
    if (pc.displayMode == 3u) return float3(roughness);
    if (pc.displayMode == 4u) return float3(metalness);
    if (pc.displayMode == 5u) return float3(clearcoat);
    if (pc.displayMode == 6u) return float3(clearcoatRoughness);
    return baseColor;
}

[shader("closesthit")]
void chitMain(inout PathPayload payload, BuiltInTriangleIntersectionAttributes attr) {
    payload.instanceId = InstanceID();

    InstanceGPU instance = instances[InstanceID()];
    MeshGPU     mesh     = meshes[instance.meshIndex];
    MaterialGPU material = materials[instance.materialIndex];

    float3x4 o2w = ObjectToWorld3x4();
    Triangle tri = fetchTriangle(mesh, PrimitiveIndex());
    Hit      hit = interpolateHit(tri, attr.barycentrics, o2w, WorldRayDirection());

    applyNormalMap(hit, material);
    float3 baseColor = getBaseColor(material, hit.uv);

    if (pc.displayMode != 0u) {
        payload.radiance = payload.throughput * debugAOV(material, hit, baseColor);
        payload.done     = true;
        return;
    }

    if (material.luminance > 0.0) {
        float3 we1       = mul(o2w, float4(tri.v1.pos - tri.v0.pos, 0.0));
        float3 we2       = mul(o2w, float4(tri.v2.pos - tri.v0.pos, 0.0));
        float  worldArea = 0.5 * length(cross(we1, we2));
        float  objArea   = 0.5 * length(cross(tri.v1.pos - tri.v0.pos, tri.v2.pos - tri.v0.pos));
        float  cosLight  = abs(dot(hit.gN, WorldRayDirection()));
        float  weight    = emitterMisWeight(mesh, objArea, worldArea, cosLight, RayTCurrent(),
                                            lightPmf(material, instance), payload.prevBrdfPdf);
        payload.radiance += payload.throughput * baseColor * material.luminance * weight;
        payload.done = true;
        return;
    }

    directLightContribution(payload, hit.P, hit.gN, hit.N, baseColor);
    sampledLightNEE(payload, hit.P, hit.gN, hit.N, baseColor);

    float  pdf;
    float3 wi = sampleDiffuse(hit.N, payload.seed, pdf);
    payload.throughput *= baseColor;
    payload.origin      = movedPoint(hit.P, hit.gN);
    payload.direction   = wi;
    payload.prevBrdfPdf = pdf;
    payload.done        = false;
}
