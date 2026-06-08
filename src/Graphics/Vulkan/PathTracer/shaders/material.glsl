#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

#include "utils.glsl"

void createONB(in vec3 N, out vec3 tangent, out vec3 bitangent) {
    float sign = copysign(1.0, N.z);
    float a = -1.0 / (sign + N.z);
    float b = N.x * N.y * a;
    tangent = vec3(1.0 + sign * N.x * N.x * a, sign * b, -sign * N.x);
    bitangent = vec3(b, sign + N.y * N.y * a, -N.y);
}

vec3 cosineSample(inout uint seed) {
    const float u = rand01(seed);
    const float v = rand01(seed);
    const float r = sqrt(u);
    float azimuth = v * 2 * M_PI;
    return vec3(r * cos(azimuth), r * sin(azimuth), sqrt(1 - u));
}

vec3 lambertianBRDF(vec3 albedo) {
    return albedo * vec3(M_1_PI);
}
float lambertianPDF(const vec3 localSample) {
    return localSample.z * M_1_PI;
}

struct Scatter {
    vec3 wi, brdf;
    float pdf;
};

Scatter lambertianScatter(inout uint seed, vec3 albedo, const vec3 N, const vec3 wo) {
    Scatter scatter;
    vec3 tangent, bitangent;
    createONB(N, tangent, bitangent);
    vec3 localSample = cosineSample(seed);
    scatter.wi = localSample.x * tangent + localSample.y * bitangent + localSample.z * N;

    scatter.pdf = lambertianPDF(localSample);
    scatter.brdf = lambertianBRDF(albedo);
    return scatter;
}

#endif

