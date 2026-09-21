import display;

groupshared uint sharedBins[HISTOGRAM_BINS];

uint luminanceBin(float lum) {
    if (lum < 1e-5) return 0u;
    float logLum = saturate((log2(lum) - pc.minLogLuminance) / max(pc.logLuminanceRange, 1e-3));
    return uint(logLum * 254.0 + 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void histogramMain(uint3 tid : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex) {
    sharedBins[groupIndex] = 0u;
    GroupMemoryBarrierWithGroupSync();

    if (tid.x < pc.renderWidth && tid.y < pc.renderHeight) {
        float3 color = tracerImage[int2(tid.xy)].rgb;
        InterlockedAdd(sharedBins[luminanceBin(luminance(color))], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    uint count = sharedBins[groupIndex];
    if (count > 0u) InterlockedAdd(exposureBuffer[groupIndex], count);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void exposureMain(uint groupIndex : SV_GroupIndex) {
    uint blackCount = exposureBuffer[0];
    sharedBins[groupIndex] = exposureBuffer[groupIndex] * groupIndex;
    GroupMemoryBarrierWithGroupSync();

    for (uint cutoff = HISTOGRAM_BINS >> 1u; cutoff > 0u; cutoff >>= 1u) {
        if (groupIndex < cutoff) sharedBins[groupIndex] += sharedBins[groupIndex + cutoff];
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex != 0u) return;

    float pixels   = float(max(pc.renderWidth * pc.renderHeight, 1u));
    float lit      = max(pixels - float(blackCount), 1.0);
    float weighted = max(float(sharedBins[0]) / lit - 1.0, 0.0);
    float avgLum   = exp2(saturate(weighted / 254.0) * pc.logLuminanceRange + pc.minLogLuminance);

    float target = MIDDLE_GREY / max(avgLum, 1e-5);
    float prev   = asfloat(exposureBuffer[EXPOSURE_SLOT]);

    float smoothed = target;
    if (prev > 0.0 && !isinf(prev) && !isnan(prev)) {
        float blend = saturate(1.0 - exp(-max(pc.deltaTime, 0.0) * ADAPTATION_SPEED));
        smoothed = lerp(prev, target, blend);
    }

    exposureBuffer[EXPOSURE_SLOT]      = asuint(clamp(smoothed, 1e-4, 1e4));
    exposureBuffer[AVG_LUMINANCE_SLOT] = asuint(avgLum);
}
