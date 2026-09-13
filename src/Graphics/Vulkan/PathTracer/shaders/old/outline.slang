[[vk::image_format("rgba16f")]]                       // format of the HDR path-traced input image
[[vk::binding(0)]]                                    // binding 0
RWTexture2D<float4> tracerImage;                      // input: accumulated HDR radiance from the ray tracer

[[vk::image_format("r32ui")]]                         // format of the instance-id image
[[vk::binding(1)]]                                    // binding 1
RWTexture2D<uint> instanceIdImage;                    // input: per-pixel instance id (0 = background)

[[vk::image_format("rgba16f")]]                       // format of the final output image
[[vk::binding(2)]]                                    // binding 2
RWTexture2D<float4> outputImage;                      // output: tonemapped image with selection outline

struct PostprocessConstants {                         // push constants for the postprocess pass
    float exposure;                                   // exposure multiplier applied before tonemapping
    uint  tonemap;                                    // 1 = apply ACES tonemap, 0 = linear
    uint  displayMode;                                // 0 = beauty (tonemap applied); nonzero = raw AOV passthrough
    uint  renderScale;                                // integer upscale factor from render to output resolution
};                                                    // end PostprocessConstants
[[vk::push_constant]]                                 // push-constant block
PostprocessConstants pc;                              // the constants

float3 acesToneMap(float3 x) {                        // Narkowicz ACES filmic tonemap approximation
    const float a = 2.51;                             // curve coefficient a
    const float b = 0.03;                             // curve coefficient b
    const float c = 2.43;                             // curve coefficient c
    const float d = 0.59;                             // curve coefficient d
    const float e = 0.14;                             // curve coefficient e
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e)); // rational tonemap curve, clamped to [0,1]
}                                                     // end acesToneMap

float4 outline(int2 pixel, float3 baseColor, const int thickness) { // draw a selection outline around the hovered/selected instance
    uint instanceId = instanceIdImage[pixel];         // id at this pixel
    uint2 size;                                       // image size holder
    instanceIdImage.GetDimensions(size.x, size.y);    // query image dimensions for bounds checks
    if (instanceId == 0) {                            // background pixel?
        return float4(baseColor, 1);                  // no outline, pass the color through
    }                                                 // end background check
    bool edge = false;                                // whether this pixel is on a selection edge
    for (int y = -thickness; y <= thickness; ++y) {   // scan a neighborhood in Y
        for (int x = -thickness; x <= thickness; ++x) { // scan a neighborhood in X
            int2 q = pixel + int2(x, y);              // neighbor coordinate
            if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y)) // out of bounds?
                continue;                             // skip it
            if (x * x + y * y > thickness * thickness) // outside the circular brush radius?
                continue;                             // skip it
            uint id = instanceIdImage[q];             // neighbor's instance id
            if (id == 0u || id > instanceId) {        // neighbor is background or a higher id => we're at a border
                edge = true;                          // mark as edge
                break;                                // stop scanning X
            }                                         // end border test
        }                                             // end X loop
        if (edge) break;                              // stop scanning Y once an edge is found
    }                                                 // end Y loop
    if (edge) {                                       // on an edge?
        return float4(1.0, 0.3, 0.0, 1.0);            // draw the orange outline color
    }                                                 // end edge branch
    return float4(baseColor, 1);                      // interior pixel: pass the color through
}                                                     // end outline

[shader("compute")]                                   // compute stage
[numthreads(16, 16, 1)]                               // 16x16 thread tiles
void main(uint3 tid : SV_DispatchThreadID) {          // one invocation per output pixel
    int2 pixel = int2(tid.xy);                        // output pixel coordinate
    int  scale = int(max(pc.renderScale, 1u));        // upscale factor (at least 1)
    float3 imageColor = tracerImage[pixel / scale].rgb; // nearest-sample the (possibly lower-res) tracer image
    if (pc.displayMode == 0u) {                       // beauty mode?
        imageColor *= pc.exposure;                    // apply exposure
        if (pc.tonemap != 0u)                         // tonemapping enabled?
            imageColor = acesToneMap(imageColor);     // apply ACES curve
    }                                                 // end beauty branch (AOVs pass through untouched)
    outputImage[pixel] = outline(pixel, imageColor, 4); // composite the selection outline and write the final pixel
}                                                     // end main
