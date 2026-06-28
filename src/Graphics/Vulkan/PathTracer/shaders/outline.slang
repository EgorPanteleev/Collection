[[vk::image_format("rgba16f")]]
[[vk::binding(0)]]
RWTexture2D<float4> tracerImage;

[[vk::image_format("r32ui")]]
[[vk::binding(1)]]
RWTexture2D<uint> instanceIdImage;

[[vk::image_format("rgba16f")]]
[[vk::binding(2)]]
RWTexture2D<float4> outputImage;

struct PostprocessConstants {
    float exposure;
    uint  tonemap;
    uint  displayMode;
};
[[vk::push_constant]]
PostprocessConstants pc;

float3 acesToneMap(float3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

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
    int2 pixel = int2(tid.xy);
    float3 imageColor = tracerImage[pixel].rgb;
    if (pc.displayMode == 0u) {
        imageColor *= pc.exposure;
        if (pc.tonemap != 0u)
            imageColor = acesToneMap(imageColor);
    }
    outputImage[pixel] = outline(pixel, imageColor, 4);
}
