module display;

public static const uint TONEMAP_OFF       = 0u;
public static const uint TONEMAP_ACES      = 1u;
public static const uint TONEMAP_NEUTRAL   = 2u;
public static const uint TONEMAP_AGX       = 3u;
public static const uint TONEMAP_FROSTBITE = 4u;

public static const uint HISTOGRAM_BINS     = 256u;
public static const uint EXPOSURE_SLOT      = 256u;
public static const uint AVG_LUMINANCE_SLOT = 257u;
public static const uint EXPOSURE_WORDS     = 258u;

public static const float SDR_WHITE_POINT   = 4.0;
public static const float MIDDLE_GREY       = 0.18;
public static const float ADAPTATION_SPEED  = 2.5;

public struct PostprocessConstants {
    public float exposure;
    public uint  tonemapMode;
    public uint  displayMode;
    public uint  renderScale;
    public uint  autoExposure;
    public float deltaTime;
    public uint  renderWidth;
    public uint  renderHeight;
    public float minLogLuminance;
    public float logLuminanceRange;
};
[[vk::push_constant]] public PostprocessConstants pc;

[[vk::image_format("rgba32f")]] [[vk::binding(0)]] public RWTexture2D<float4> tracerImage;
[[vk::image_format("r32ui")]]   [[vk::binding(1)]] public RWTexture2D<uint>   instanceIdImage;
[[vk::image_format("rgba16f")]] [[vk::binding(2)]] public RWTexture2D<float4> outputImage;
[[vk::binding(3)]] public RWStructuredBuffer<uint> exposureBuffer;

static const float3x3 SRGB_TO_REC2020 = float3x3(
    0.6274, 0.3293, 0.0433,
    0.0691, 0.9195, 0.0113,
    0.0164, 0.0880, 0.8956);

static const float3x3 REC2020_TO_SRGB = float3x3(
     1.6605, -0.5876, -0.0728,
    -0.1246,  1.1329, -0.0083,
    -0.0182, -0.1006,  1.1187);

public float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

public float currentExposure() {
    if (pc.autoExposure == 0u) return pc.exposure;
    return max(asfloat(exposureBuffer[EXPOSURE_SLOT]), 1e-5);
}

float3 acesToneMap(float3 color) {
    const float3x3 inputMat = float3x3(
        0.59719, 0.35458, 0.04823,
        0.07600, 0.90834, 0.01566,
        0.02840, 0.13383, 0.83777);
    const float3x3 outputMat = float3x3(
         1.60475, -0.53108, -0.07367,
        -0.10208,  1.10813, -0.00605,
        -0.00327, -0.07276,  1.07602);

    float3 v = mul(inputMat, color);
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return saturate(mul(outputMat, a / b));
}

float3 neutralToneMap(float3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation     = 0.15;

    float x      = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d       = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return lerp(color, float3(newPeak), g);
}

float3 agxContrast(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4
         - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

float3 agxToneMap(float3 color) {
    const float3x3 insetMat = float3x3(
        0.856627153315983,  0.0951212405381588, 0.0482516061458583,
        0.137318972929847,  0.761241990602591,  0.101439036467562,
        0.11189821299995,   0.0767994186031903, 0.811302368396859);
    const float3x3 outsetMat = float3x3(
         1.1271005818144368, -0.11060664309660323, -0.016493938717834573,
        -0.1413297634984383,  1.157823702216272,   -0.016493938717834257,
        -0.14132976349843826, -0.11060664309660294,  1.2519364065950405);
    const float minEv = -12.47393;
    const float maxEv = 4.026069;

    float3 c = mul(SRGB_TO_REC2020, max(color, 0.0));
    c = mul(insetMat, c);
    c = log2(max(c, 1e-10));
    c = saturate((c - minEv) / (maxEv - minEv));
    c = agxContrast(c);
    c = mul(outsetMat, c);
    c = pow(max(c, 0.0), 2.2);
    return saturate(mul(REC2020_TO_SRGB, c));
}

float extendedReinhard(float x, float whitePoint) {
    float w2 = max(whitePoint * whitePoint, 1e-4);
    return x * (1.0 + x / w2) / (1.0 + x);
}

float3 frostbiteToneMap(float3 color, float whitePoint) {
    float lum = luminance(color);
    if (lum <= 1e-6) return max(color, 0.0);

    float  curvedLum  = extendedReinhard(lum, whitePoint);
    float3 chroma     = color * (curvedLum / lum);
    float3 perChannel = float3(extendedReinhard(color.r, whitePoint),
                               extendedReinhard(color.g, whitePoint),
                               extendedReinhard(color.b, whitePoint));
    float  bleach     = saturate(curvedLum / max(whitePoint, 1.0));
    return max(lerp(chroma, perChannel, bleach * bleach), 0.0);
}

public float3 displayMap(float3 color) {
    if (pc.tonemapMode == TONEMAP_ACES)      return acesToneMap(color);
    if (pc.tonemapMode == TONEMAP_NEUTRAL)   return neutralToneMap(color);
    if (pc.tonemapMode == TONEMAP_AGX)       return agxToneMap(color);
    if (pc.tonemapMode == TONEMAP_FROSTBITE) return saturate(frostbiteToneMap(color, SDR_WHITE_POINT));
    return saturate(color);
}
