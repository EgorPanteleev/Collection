module bsdf;

import common;

public void createONB(float3 N, out float3 tangent, out float3 bitangent) {
    float s = copySign(1.0, N.z);
    float a = -1.0 / (s + N.z);
    float b = N.x * N.y * a;
    tangent   = float3(1.0 + s * N.x * N.x * a, s * b, -s * N.x);
    bitangent = float3(b, s + N.y * N.y * a, -N.y);
}

public float3 cosineSampleLocal(inout uint seed) {
    float u = rand01(seed);
    float v = rand01(seed);
    float r = sqrt(u);
    float azimuth = v * 2.0 * M_PI;
    return float3(r * cos(azimuth), r * sin(azimuth), sqrt(1.0 - u));
}

public float3 diffuseBrdf(float3 albedo) {
    return albedo * M_1_PI;
}

public float diffusePdf(float NdotWi) {
    return max(NdotWi, 0.0) * M_1_PI;
}

public float3 sampleDiffuse(float3 N, inout uint seed, out float pdf) {
    float3 tangent, bitangent;
    createONB(N, tangent, bitangent);
    float3 local = cosineSampleLocal(seed);
    pdf = local.z * M_1_PI;
    return local.x * tangent + local.y * bitangent + local.z * N;
}
