import display;

float4 outline(int2 pixel, float3 baseColor, const int thickness) {
    uint instanceId = instanceIdImage[pixel];
    uint2 size;
    instanceIdImage.GetDimensions(size.x, size.y);
    if (instanceId == 0) {
        return float4(baseColor, 1);
    }
    bool edge = false;
    for (int y = -thickness; y <= thickness; ++y) {
        for (int x = -thickness; x <= thickness; ++x) {
            int2 q = pixel + int2(x, y);
            if (q.x < 0 || q.y < 0 || q.x >= int(size.x) || q.y >= int(size.y))
                continue;
            if (x * x + y * y > thickness * thickness)
                continue;
            uint id = instanceIdImage[q];
            if (id == 0u || id > instanceId) {
                edge = true;
                break;
            }
        }
        if (edge) break;
    }
    if (edge) {
        return float4(1.0, 0.3, 0.0, 1.0);
    }
    return float4(baseColor, 1);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint2 size;
    outputImage.GetDimensions(size.x, size.y);
    if (tid.x >= size.x || tid.y >= size.y) return;

    int2 pixel = int2(tid.xy);
    int  scale = int(max(pc.renderScale, 1u));
    float3 imageColor = tracerImage[pixel / scale].rgb;
    if (pc.displayMode == 0u) {
        imageColor = displayMap(imageColor * currentExposure());
    }
    outputImage[pixel] = outline(pixel, imageColor, 4);
}
