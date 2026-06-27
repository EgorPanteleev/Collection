import common;
import types;
import bsdf;
import geometry;
import lighting;
import scene;

[shader("raygeneration")]
void rgenMain() {
    uint2 launchID   = DispatchRaysIndex().xy;
    uint2 launchSize = DispatchRaysDimensions().xy;
    int2  pixel      = int2(launchID);
    uint  seed       = launchID.x + launchID.y * launchSize.x + pc.frame * 16777619u;

    float3 origin, dir;
    primaryRay(launchID, launchSize, seed, origin, dir);

    PathPayload payload;
    payload.throughput  = float3(1.0);
    payload.radiance    = float3(0.0);
    payload.origin      = origin;
    payload.direction   = dir;
    payload.instanceId  = UINT_MAX;
    payload.seed        = seed;
    payload.depth       = 0;
    payload.prevBrdfPdf = -1.0;
    payload.done        = false;
    payload.insideGlass = false;

    uint firstInstanceId = UINT_MAX;
    for (uint depth = 0; depth < pc.maxDepth; ++depth) {
        payload.depth = depth;
        RayDesc ray;
        ray.Origin    = payload.origin;
        ray.TMin      = 0.0;
        ray.Direction = payload.direction;
        ray.TMax      = T_MAX;
        TraceRay(tlas, RAY_FLAG_NONE, ALL_OBJECTS, 0, 0, 0, ray, payload);

        if (depth == 0) firstInstanceId = payload.instanceId;
        if (payload.done) break;

        if (depth >= pc.minDepth) {
            float p = clamp(luminance(payload.throughput), 0.01, 1.0);
            if (rand01(payload.seed) > p) break;
            payload.throughput /= p;
        }
    }

    if (pc.frame == 0) {
        outputImage[pixel] = float4(payload.radiance, 1.0);
        return;
    }
    float3 imageColor = outputImage[pixel].rgb;
    float3 newColor   = (imageColor * float(pc.frame) + payload.radiance) / float(pc.frame + 1);
    outputImage[pixel]           = float4(newColor, 1.0);
    outputInstanceIdImage[pixel] = firstInstanceId + 1;
}

[shader("miss")]
void missMain(inout PathPayload payload) {
    if (pc.displayMode == 0u) {
        float3 contrib = payload.throughput * sampleEnvironment(WorldRayDirection());
        if (payload.depth > 0u) {
            float lum = luminance(contrib);
            if (lum > FIREFLY_CLAMP) contrib *= FIREFLY_CLAMP / lum;
        }
        payload.radiance += contrib;
    }
    payload.done = true;
}

[shader("miss")]
void shadowMissMain(inout float visible) {
    visible = 1.0;
}

[shader("closesthit")]
void chitMain(inout PathPayload payload, BuiltInTriangleIntersectionAttributes attr) {
    payload.instanceId = InstanceID();
    InstanceGPU  instance = instances[InstanceID()];
    MeshGPU     mesh     = meshes[instance.meshIndex];

    float3x4    o2w = ObjectToWorld3x4();
    Triangle    tri = fetchTriangle(mesh, PrimitiveIndex());
    HitGeometry hit = interpolateHit(tri, attr.barycentrics, o2w, WorldRayDirection());

    MaterialGPU material  = materials[instance.materialIndex];
    applyNormalMap(hit, material);
    float3       baseColor = getBaseColor(material, hit.uv);

    if (pc.displayMode != 0u && payload.depth == 0u) {
        float dmMetal = material.metalness;
        float dmRough = material.roughness;
        applyMetalRoughness(material, hit.uv, dmMetal, dmRough);
        float dmClearcoat = material.clearcoat;
        float dmClearcoatRough = material.clearcoatRoughness;
        applyClearcoat(material, hit.uv, dmClearcoat, dmClearcoatRough);
        float3 aov = baseColor;
        if (pc.displayMode == 2u) aov = hit.N * 0.5 + 0.5;
        else if (pc.displayMode == 3u) aov = float3(dmRough);
        else if (pc.displayMode == 4u) aov = float3(dmMetal);
        else if (pc.displayMode == 5u) aov = float3(dmClearcoat);
        else if (pc.displayMode == 6u) aov = float3(dmClearcoatRough);
        payload.radiance = aov;
        payload.done     = true;
        return;
    }

    if (material.luminance > 0.0) {
        float weight = 1.0;
        if (pc.nee != 0u && payload.prevBrdfPdf >= 0.0) {
            float  objArea = 0.5 * length(cross(tri.v1.pos - tri.v0.pos, tri.v2.pos - tri.v0.pos));
            float3 we1     = mul(o2w, float4(tri.v1.pos - tri.v0.pos, 0.0));
            float3 we2     = mul(o2w, float4(tri.v2.pos - tri.v0.pos, 0.0));
            weight = emitterMisWeight(mesh, objArea, we1, we2, hit.N, WorldRayDirection(),
                                      RayTCurrent(), pc.emissiveCount, payload.prevBrdfPdf);
        }
        payload.radiance += payload.throughput * material.emissionColor * material.luminance * weight;
        payload.done = true;
        return;
    }

    if (material.transmission > 0.0) {
        float pt = material.transmission;
        if (rand01(payload.seed) < pt) {
            Scatter scatter;
            float3 sigmaA = -log(max(material.attenuationColor, float3(1e-4))) * material.absorption;
            if (material.thin > 0.0) {
                scatter = sampleThinDielectric(payload.seed, hit.N, -WorldRayDirection(),
                                               material.ior, sigmaA, material.thickness);
            } else {
                if (payload.insideGlass) {
                    payload.throughput *= exp(-sigmaA * RayTCurrent());
                }
                scatter = sampleDielectric(payload.seed, hit.N, -WorldRayDirection(),
                                           material.ior, payload.insideGlass);
                if (dot(scatter.wi, hit.N) < 0.0) payload.insideGlass = !payload.insideGlass;
            }
            payload.throughput *= scatter.brdf / pt;
            payload.origin      = movedPoint(hit.P, dot(scatter.wi, hit.gN) < 0.0 ? -hit.gN : hit.gN);
            payload.direction   = scatter.wi;
            payload.prevBrdfPdf = -1.0;
            payload.done        = false;
            return;
        }
        payload.throughput /= (1.0 - pt);
    }

    float metalness = material.metalness;
    float roughness = material.roughness;
    applyMetalRoughness(material, hit.uv, metalness, roughness);
    float clearcoat = material.clearcoat;
    float clearcoatRoughness = material.clearcoatRoughness;
    applyClearcoat(material, hit.uv, clearcoat, clearcoatRoughness);
    Surface surf = makeSurface(baseColor, metalness, roughness,
                               material.ior, material.specular,
                               clearcoat, clearcoatRoughness,
                               hit.N, -WorldRayDirection());
    Scatter scatter = sampleBSDF(payload.seed, surf);

    directLightContribution(payload, hit.P, hit.gN, surf);
    nextEventEstimation(payload, hit.P, hit.gN, surf);

    if (scatter.pdf <= PDF_EPS) {
        payload.done = true;
        return;
    }

    payload.throughput *= scatter.brdf * max(dot(scatter.wi, hit.N), 0.0) / scatter.pdf;
    payload.origin      = movedPoint(hit.P, hit.gN);
    payload.direction   = scatter.wi;
    payload.prevBrdfPdf = scatter.pdf;
    payload.done        = false;
}
