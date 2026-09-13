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
    float3 dir  = WorldRayDirection();
    float  misW = 1.0;
    if (payload.prevBrdfPdf >= 0.0) {
        float ep = envPdf(dir);
        if (ep > 0.0) misW = powerHeuristic(payload.prevBrdfPdf, ep);
    }
    payload.radiance += payload.throughput * sampleEnvironment(dir) * misW;
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

    if (material.transmission > 0.0) {
        if (rand01(payload.seed) < material.transmission) {
            float3 view = -WorldRayDirection();
            float  eta  = hit.frontFace ? (1.0 / material.ior) : material.ior;
            Scatter s   = makeDielectric(eta).sampleF(view, hit.N, payload.seed);
            payload.throughput *= s.f / material.transmission;
            payload.origin      = movedPoint(hit.P, dot(s.wi, hit.gN) > 0.0 ? hit.gN : -hit.gN);
            payload.direction   = s.wi;
            payload.prevBrdfPdf = -1.0;
            payload.done        = false;
            return;
        }
        payload.throughput /= (1.0 - material.transmission);
    }

    float metalness = material.metalness;
    float roughness = material.roughness;
    applyMetalRoughness(material, hit.uv, metalness, roughness);
    float clearcoat = material.clearcoat;
    float clearcoatRoughness = material.clearcoatRoughness;
    applyClearcoat(material, hit.uv, clearcoat, clearcoatRoughness);

    float3 wo = -WorldRayDirection();
    BSDF bsdf = makeBSDF(hit.N, hit.tangent, baseColor, metalness, roughness,
                         material.anisotropy, clearcoat, clearcoatRoughness);

    directLightContribution(payload, bsdf, wo, hit.P, hit.gN);
    sampledLightNEE(payload, bsdf, wo, hit.P, hit.gN);
    sampledEnvNEE(payload, bsdf, wo, hit.P, hit.gN);

    Scatter s = bsdf.sampleF(wo, payload.seed);
    if (s.pdf <= 0.0) {
        payload.done = true;
        return;
    }
    payload.throughput *= s.f * abs(dot(s.wi, hit.N)) / s.pdf;
    payload.origin      = movedPoint(hit.P, hit.gN);
    payload.direction   = s.wi;
    payload.prevBrdfPdf = s.pdf;
    payload.done        = false;
}
