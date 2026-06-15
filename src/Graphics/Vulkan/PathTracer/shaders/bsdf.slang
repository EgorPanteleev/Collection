module bsdf;

import common;

public struct Scatter {
    public float3 wi;
    public float3 brdf;
    public float  pdf;
};

public void createONB(float3 N, out float3 tangent, out float3 bitangent) {
    float s = copySign(1.0, N.z);
    float a = -1.0 / (s + N.z);
    float b = N.x * N.y * a;
    tangent   = float3(1.0 + s * N.x * N.x * a, s * b, -s * N.x);
    bitangent = float3(b, s + N.y * N.y * a, -N.y);
}

public float3 cosineSample(inout uint seed) {
    float u = rand01(seed);
    float v = rand01(seed);
    float r = sqrt(u);
    float azimuth = v * 2.0 * M_PI;
    return float3(r * cos(azimuth), r * sin(azimuth), sqrt(1.0 - u));
}

public float3 lambertianBRDF(float3 albedo) {
    return albedo * M_1_PI;
}

public float lambertianPDF(float3 localSample) {
    return localSample.z * M_1_PI;
}

public Scatter lambertianScatter(inout uint seed, float3 albedo, float3 N, float3 wo) {
    Scatter scatter;
    float3 tangent, bitangent;
    createONB(N, tangent, bitangent);
    float3 localSample = cosineSample(seed);
    scatter.wi   = localSample.x * tangent + localSample.y * bitangent + localSample.z * N;
    scatter.pdf  = lambertianPDF(localSample);
    scatter.brdf = lambertianBRDF(albedo);
    return scatter;
}
