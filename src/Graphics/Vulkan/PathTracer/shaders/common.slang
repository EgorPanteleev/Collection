module common;

public static const float FLT_MAX  = 3.402823e38;
public static const uint  UINT_MAX = 4294967295u;
public static const float M_PI     = 3.14159265359;
public static const float M_1_PI   = 0.3183098861837907;

public static const uint  ALL_OBJECTS = 0xFF;
public static const float T_MIN       = 0.0;
public static const float T_MAX       = 1e10;

public static const float ORIGIN_THRESHOLD = 0.03125;
public static const float FLOAT_SCALE      = 1.0 / 65536.0;
public static const float INT_SCALE        = 256.0;
public static const float SHADOW_EPS       = 1e-3;

public static const float PDF_EPS   = 1e-4;
public static const float NEE_CLAMP = 50.0;
public static const float FIREFLY_CLAMP = 10.0;

public struct PushConstants {
    public uint frame;
    public uint spp;
    public uint minDepth;
    public uint maxDepth;
    public uint displayMode;
    public uint nee;
    public uint emissiveCount;
    public uint skyboxIndex;
    public float    envIntegral;
    public uint     envNee;
    public float    aperture;
    public float    focusDistance;
    public uint64_t envMarginalCdfAddr;
    public uint64_t envCondCdfAddr;
    public uint64_t envCondFuncAddr;
    public float3   skyColor;
};
[[vk::push_constant]] public PushConstants pc;

public uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}

public uint owenScramble(uint x, uint seed) {
    x = reversebits(x);
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    x = reversebits(x);
    return x;
}

public float rand01(inout uint seed) {
    seed = pcg(seed);
    uint keyIndex = seed;
    seed = pcg(seed);
    uint keyValue = seed;
    uint index = owenScramble(pc.frame, keyIndex);
    uint value = owenScramble(reversebits(index), keyValue);
    return float(value >> 8) * (1.0 / 16777216.0);
}

public float luminance(float3 c) {
    return dot(c, float3(0.2126, 0.7152, 0.0722));
}

public float copySign(float x, float y) {
    return abs(x) * (y < 0.0 ? -1.0 : 1.0);
}

public float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    float denom = a2 + b2;
    return denom > 0.0 ? a2 / denom : 0.0;
}

public float3x3 inverse3x3(float3x3 m) {
    float a00 = m[0][0], a01 = m[0][1], a02 = m[0][2];
    float a10 = m[1][0], a11 = m[1][1], a12 = m[1][2];
    float a20 = m[2][0], a21 = m[2][1], a22 = m[2][2];

    float det = a00 * (a11 * a22 - a12 * a21)
              - a01 * (a10 * a22 - a12 * a20)
              + a02 * (a10 * a21 - a11 * a20);
    float invDet = 1.0 / det;

    float3x3 inv;
    inv[0][0] =  (a11 * a22 - a12 * a21) * invDet;
    inv[0][1] = -(a01 * a22 - a02 * a21) * invDet;
    inv[0][2] =  (a01 * a12 - a02 * a11) * invDet;
    inv[1][0] = -(a10 * a22 - a12 * a20) * invDet;
    inv[1][1] =  (a00 * a22 - a02 * a20) * invDet;
    inv[1][2] = -(a00 * a12 - a02 * a10) * invDet;
    inv[2][0] =  (a10 * a21 - a11 * a20) * invDet;
    inv[2][1] = -(a00 * a21 - a01 * a20) * invDet;
    inv[2][2] =  (a00 * a11 - a01 * a10) * invDet;
    return inv;
}

public float3 offsetRay(float3 p, float3 n) {
    int3 of_i = int3(int(INT_SCALE * n.x), int(INT_SCALE * n.y), int(INT_SCALE * n.z));
    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return float3(
        abs(p.x) < ORIGIN_THRESHOLD ? p.x + FLOAT_SCALE * n.x : p_i.x,
        abs(p.y) < ORIGIN_THRESHOLD ? p.y + FLOAT_SCALE * n.y : p_i.y,
        abs(p.z) < ORIGIN_THRESHOLD ? p.z + FLOAT_SCALE * n.z : p_i.z);
}

public float3 movedPoint(float3 p, float3 gn) {
    return offsetRay(p, gn);
}
