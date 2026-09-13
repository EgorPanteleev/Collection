import common;                                        // pc, RNG, constants, clamps
import types;                                         // PathPayload
import bsdf;                                          // Surface, Scatter, sampleBSDF/evalBSDF, dielectric sampling
import geometry;                                      // Triangle/HitGeometry + fetch/interpolate
import lighting;                                      // emissive-mesh sampling + MIS weight
import scene;                                         // bindings, base color, NEE, env sampling, shadow rays

[shader("raygeneration")]                             // ray generation: integrate one pixel over spp samples and bounces
void rgenMain() {                                     // SBT raygen entry "rgenMain"
    uint2 launchID   = DispatchRaysIndex().xy;        // pixel coordinate
    uint2 launchSize = DispatchRaysDimensions().xy;   // render size
    int2  pixel      = int2(launchID);                // signed coord for image writes
    uint  pixelId    = launchID.x + launchID.y * launchSize.x; // flat pixel index

    uint   sampleCount    = max(pc.spp, 1u);          // samples per pixel this frame
    float3 radianceSum    = float3(0.0);              // per-frame radiance accumulator
    uint   firstInstanceId = UINT_MAX;                // instance id of the first primary hit (selection buffer)

    for (uint s = 0; s < sampleCount; ++s) {          // loop over pixel samples
        uint seed = pcg(pixelId + s * 0x9e3779b9u);   // per-sample RNG seed
        float3 origin, dir;                           // primary ray outputs
        primaryRay(launchID, launchSize, seed, origin, dir); // jittered camera ray (with DoF)

        PathPayload payload;                          // path state
        payload.throughput  = float3(1.0);            // full throughput at start
        payload.radiance    = float3(0.0);            // no radiance yet
        payload.origin      = origin;                 // ray origin
        payload.direction   = dir;                    // ray direction
        payload.instanceId  = UINT_MAX;               // nothing hit yet
        payload.seed        = seed;                   // RNG state
        payload.depth       = 0;                      // start depth
        payload.prevBrdfPdf = -1.0;                   // no prior BSDF pdf (first bounce has no MIS partner)
        payload.done        = false;                  // not finished
        payload.insideGlass = false;                  // start in vacuum/air

        for (uint depth = 0; depth < pc.maxDepth; ++depth) { // bounce loop
            payload.depth = depth;                    // record current depth
            RayDesc ray;                              // ray for this segment
            ray.Origin    = payload.origin;           // segment origin (updated by the hit shader)
            ray.TMin      = 0.0;                       // start distance
            ray.Direction = payload.direction;        // segment direction
            ray.TMax      = T_MAX;                     // far distance
            TraceRay(tlas, RAY_FLAG_NONE, ALL_OBJECTS, 0, 0, 0, ray, payload); // trace; closest-hit/miss update payload

            if (depth == 0 && s == 0) firstInstanceId = payload.instanceId; // capture primary instance id once
            if (payload.done) break;                  // path terminated (miss, emitter, or absorbed)

            if (depth >= pc.minDepth) {               // after minDepth bounces, apply Russian roulette
                float p = clamp(luminance(payload.throughput), 0.01, 1.0); // survival probability from throughput
                if (rand01(payload.seed) > p) break;  // terminate this path
                payload.throughput /= p;              // compensate survivors to stay unbiased
            }                                         // end Russian roulette
        }                                             // end bounce loop

        radianceSum += payload.radiance;              // add this sample's radiance
    }                                                 // end sample loop

    float3 radiance = radianceSum / float(sampleCount); // average over samples
    if (any(isnan(radiance)) || any(isinf(radiance))) radiance = float3(0.0); // NaN/Inf guard

    outputInstanceIdImage[pixel] = firstInstanceId + 1; // write selection id (+1 so 0 = background)

    if (pc.frame == 0) {                              // first accumulation frame
        outputImage[pixel] = float4(radiance, 1.0);   // seed the accumulator
        return;                                        // done
    }                                                 // end first-frame branch
    float3 imageColor = outputImage[pixel].rgb;       // previous accumulated color
    float3 newColor   = (imageColor * float(pc.frame) + radiance) / float(pc.frame + 1); // progressive average
    outputImage[pixel] = float4(newColor, 1.0);       // store the new average
}                                                     // end rgenMain

[shader("miss")]                                      // miss shader for camera/indirect rays
void missMain(inout PathPayload payload) {            // SBT miss 0 "missMain"
    if (pc.displayMode == 0u) {                       // only accumulate environment light in beauty mode
        float3 dir  = WorldRayDirection();            // ray direction that missed
        float  misW = 1.0;                            // MIS weight (default full)
        if (payload.prevBrdfPdf >= 0.0) {             // previous bounce had a samplable pdf?
            float ep = envPdf(dir);                   // env importance-sampling pdf for this direction
            if (ep > 0.0) misW = powerHeuristic(payload.prevBrdfPdf, ep); // MIS between BSDF and env sampling
        }                                             // end MIS check
        float3 contrib = payload.throughput * sampleEnvironment(dir) * misW; // weighted environment contribution
        if (payload.depth > 0u) {                     // on indirect bounces
            float lum = luminance(contrib);           // luminance of the contribution
            if (lum > FIREFLY_CLAMP) contrib *= FIREFLY_CLAMP / lum; // clamp fireflies
        }                                             // end firefly clamp
        payload.radiance += contrib;                  // add environment radiance
    }                                                 // end beauty-mode branch
    payload.done = true;                              // the path ends at the environment
}                                                     // end missMain

[shader("miss")]                                      // miss shader for shadow rays
void shadowMissMain(inout float visible) {            // SBT miss 1 "shadowMissMain"
    visible = 1.0;                                    // shadow ray reached the light (unoccluded)
}                                                     // end shadowMissMain

[shader("closesthit")]                                // closest-hit: full surface shading + path continuation
void chitMain(inout PathPayload payload, BuiltInTriangleIntersectionAttributes attr) { // SBT hit "chitMain"
    payload.instanceId = InstanceID();                // record hit instance
    InstanceGPU  instance = instances[InstanceID()];  // instance record
    MeshGPU     mesh     = meshes[instance.meshIndex]; // mesh record

    float3x4    o2w = ObjectToWorld3x4();             // object->world transform
    Triangle    tri = fetchTriangle(mesh, PrimitiveIndex()); // hit triangle vertices
    HitGeometry hit = interpolateHit(tri, attr.barycentrics, o2w, WorldRayDirection()); // interpolate hit attributes

    MaterialGPU material = materials[instance.materialIndex]; // material record

    if (pc.displayMode != 0u && payload.depth == 0u) { // AOV debug views use only the primary hit
        applyNormalMap(hit, material);                // perturb normal for the normal AOV
        float3 baseColor = getBaseColor(material, hit.uv); // albedo
        float dmMetal = material.metalness;           // metalness base
        float dmRough = material.roughness;           // roughness base
        applyMetalRoughness(material, hit.uv, dmMetal, dmRough); // fold metal/rough texture
        float dmClearcoat = material.clearcoat;       // clearcoat base
        float dmClearcoatRough = material.clearcoatRoughness; // clearcoat roughness base
        applyClearcoat(material, hit.uv, dmClearcoat, dmClearcoatRough); // fold clearcoat textures
        float3 aov = baseColor;                       // default AOV = base color
        if (pc.displayMode == 2u) aov = hit.N * 0.5 + 0.5; // normals
        else if (pc.displayMode == 3u) aov = float3(dmRough); // roughness
        else if (pc.displayMode == 4u) aov = float3(dmMetal); // metalness
        else if (pc.displayMode == 5u) aov = float3(dmClearcoat); // clearcoat
        else if (pc.displayMode == 6u) aov = float3(dmClearcoatRough); // clearcoat roughness
        payload.radiance = aov;                       // output AOV directly
        payload.done     = true;                      // stop (debug view is single-bounce)
        return;                                       // done
    }                                                 // end AOV branch

    if (material.opacity < 1.0 && rand01(payload.seed) >= material.opacity) { // stochastic alpha cutout: treat as transparent
        float3 dir = WorldRayDirection();             // keep going straight through
        payload.origin      = movedPoint(hit.P, dot(dir, hit.gN) < 0.0 ? -hit.gN : hit.gN); // offset past the surface
        payload.direction   = dir;                    // same direction
        payload.prevBrdfPdf = -1.0;                   // no BSDF event (pass-through)
        payload.done        = false;                  // continue path
        return;                                       // done
    }                                                 // end alpha cutout

    float3 baseColor = getBaseColor(material, hit.uv); // albedo for shading below

    if (material.luminance > 0.0) {                   // surface is an emitter?
        float weight = 1.0;                           // MIS weight (default full)
        if (pc.nee != 0u && payload.prevBrdfPdf >= 0.0) { // if NEE is on and we arrived via a BSDF bounce
            float  objArea = 0.5 * length(cross(tri.v1.pos - tri.v0.pos, tri.v2.pos - tri.v0.pos)); // object-space area
            float3 we1     = mul(o2w, float4(tri.v1.pos - tri.v0.pos, 0.0)); // world edge 1
            float3 we2     = mul(o2w, float4(tri.v2.pos - tri.v0.pos, 0.0)); // world edge 2
            weight = emitterMisWeight(mesh, objArea, we1, we2, hit.N, WorldRayDirection(), // MIS weight vs light sampling
                                      RayTCurrent(), pc.emissiveCount, payload.prevBrdfPdf);
        }                                             // end MIS block
        payload.radiance += payload.throughput * baseColor * material.luminance * weight; // add emitted radiance
        payload.done = true;                          // emitters terminate the path
        return;                                       // done
    }                                                 // end emitter branch

    applyNormalMap(hit, material);                    // apply normal map before shading

    if (material.translucency > 0.0) {                // thin translucent (diffuse transmission) material?
        float3 kr  = baseColor;                       // reflection albedo
        float3 kt  = material.translucency * baseColor; // transmission albedo
        float  nrm = max(luminance(kr + kt), 1.0);    // energy normalizer (keep <= 1)
        kr /= nrm; kt /= nrm;                         // normalize the two lobes
        float  pr  = luminance(kr);                   // reflect luminance
        float  pt  = luminance(kt);                   // transmit luminance
        float  sum = pr + pt;                         // total
        if (sum <= 0.0) { payload.done = true; return; } // nothing to scatter
        float probR = pr / sum;                       // probability of reflecting

        float3 Nface = dot(hit.N, -WorldRayDirection()) < 0.0 ? -hit.N : hit.N; // normal facing the viewer

        if (directLight.intensity > 0.0) {            // direct sun/dir-light contribution
            float3 L    = normalize(-directLight.dir); // direction to the light
            float  side = dot(L, Nface);              // which side the light is on
            float3 f    = side > 0.0 ? kr * M_1_PI * side : kt * M_1_PI * (-side); // reflect or transmit Lambertian
            if (any(f > float3(0.0))) {               // any contribution?
                float vis = traceShadow(hit.P, side > 0.0 ? hit.gN : -hit.gN, L, T_MAX); // shadow test toward the light
                payload.radiance += payload.throughput * f * directLight.intensity * vis; // accumulate direct light
            }                                         // end contribution test
        }                                             // end direct light

        float3 t, b, wi;                              // basis + sampled direction
        if (rand01(payload.seed) < probR) {           // sample the reflection lobe
            createONB(Nface, t, b);                    // basis around the front normal
            float3 local = cosineSample(payload.seed); // cosine sample
            wi = local.x * t + local.y * b + local.z * Nface; // to world space
            payload.throughput *= kr / probR;          // weight by reflect albedo / prob
        } else {                                      // sample the transmission lobe
            float3 Nopp = -Nface;                     // opposite hemisphere
            createONB(Nopp, t, b);                     // basis around it
            float3 local = cosineSample(payload.seed); // cosine sample
            wi = local.x * t + local.y * b + local.z * Nopp; // to world space
            payload.throughput *= kt / (1.0 - probR);  // weight by transmit albedo / prob
        }                                             // end lobe choice
        payload.origin      = movedPoint(hit.P, dot(wi, hit.gN) < 0.0 ? -hit.gN : hit.gN); // offset along the chosen side
        payload.direction   = normalize(wi);          // next direction
        payload.prevBrdfPdf = -1.0;                   // treated as non-MIS (diffuse transmission)
        payload.done        = false;                  // continue
        return;                                       // done
    }                                                 // end translucency branch

    if (material.transmission > 0.0) {                // specular dielectric (glass) transmission?
        float pt = material.transmission;             // probability of a glass interaction
        if (rand01(payload.seed) < pt) {              // do the glass event
            float3 sigmaA = -log(max(material.absorption, float3(1e-4))); // Beer-Lambert absorption coefficient
            if (payload.insideGlass) {                // if traveling inside the medium
                payload.throughput *= exp(-sigmaA * RayTCurrent()); // attenuate over the traveled distance
            }                                         // end absorption
            Scatter scatter = sampleDielectric(payload.seed, hit.N, -WorldRayDirection(), // sample reflect/refract
                                               material.ior, payload.insideGlass);
            if (dot(scatter.wi, hit.N) < 0.0) {       // direction went through the surface (refraction)
                payload.insideGlass = !payload.insideGlass; // toggle inside/outside state
                payload.throughput *= baseColor;       // tint transmitted light by base color
            }                                         // end refraction handling
            payload.throughput *= scatter.brdf / pt;   // apply specular value and undo the selection probability
            payload.origin      = movedPoint(hit.P, dot(scatter.wi, hit.gN) < 0.0 ? -hit.gN : hit.gN); // offset along the chosen side
            payload.direction   = scatter.wi;          // next direction
            payload.prevBrdfPdf = -1.0;                // delta event -> no MIS
            payload.done        = false;               // continue
            return;                                    // done
        }                                             // end glass event
        payload.throughput /= (1.0 - pt);             // not glass this time: undo the selection probability
    }                                                 // end transmission branch

    float metalness = material.metalness;             // scalar metalness
    float roughness = material.roughness;             // scalar roughness
    applyMetalRoughness(material, hit.uv, metalness, roughness); // fold metal/rough texture
    float clearcoat = material.clearcoat;             // scalar clearcoat
    float clearcoatRoughness = material.clearcoatRoughness; // scalar clearcoat roughness
    applyClearcoat(material, hit.uv, clearcoat, clearcoatRoughness); // fold clearcoat textures
    Surface surf = makeSurface(baseColor, metalness, roughness, // build the shading surface
                               material.ior, material.specular,
                               clearcoat, clearcoatRoughness,
                               material.anisotropy, material.sheen,
                               hit.tangent, hit.N, -WorldRayDirection());
    Scatter scatter = sampleBSDF(payload.seed, surf); // sample a BSDF direction for the indirect bounce

    directLightContribution(payload, hit.P, hit.gN, surf); // NEE: sample the directional light
    sampledLightNEE(payload, hit.P, hit.gN, surf);    // NEE: sample emissive meshes and/or the environment

    if (scatter.pdf <= PDF_EPS) {                     // degenerate BSDF sample?
        payload.done = true;                          // stop the path
        return;                                       // done
    }                                                 // end pdf guard

    payload.throughput *= scatter.brdf * max(dot(scatter.wi, hit.N), 0.0) / scatter.pdf; // Monte Carlo BSDF weight
    payload.origin      = movedPoint(hit.P, hit.gN);  // offset the next ray off the surface
    payload.direction   = scatter.wi;                 // continue along the sampled direction
    payload.prevBrdfPdf = scatter.pdf;                // remember pdf for MIS at the next emitter/env hit
    payload.done        = false;                      // keep bouncing
}                                                     // end chitMain
