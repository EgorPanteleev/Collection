#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

#include "utils.glsl"

//void createONB(in vec3 n, out vec3 b1, out vec3 b2){
//    vec3 up = abs(n.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
//
//    b1 = normalize(cross(up, n));
//    b2 = cross(n, b1);
//}

void createONB(in vec3 N, out vec3 tangent, out vec3 bitangent){
    if (abs(N.x) > abs(N.z))
        tangent = normalize(vec3(-N.y, N.x, 0));
    else
        tangent = normalize(vec3(0, -N.z, N.y));
    bitangent = cross(N, tangent);
}

vec3 cosineSample(uint seed) {
    const float u = rand01(seed);
    const float v = rand01(seed + 1);
    const float r = sqrt(u);
    float azimuth = v * 2 * M_PI;
    return vec3(r * cos(azimuth), r * sin(azimuth), sqrt(1 - u));
}

vec3 lambertianBRDF(vec3 albedo) {
    return albedo * vec3(M_1_PI);
}
float lambertianPDF(const vec3 N, const vec3 wi) {
    float theta = max(dot(N, wi), 0.0);
    return theta * M_1_PI;
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

    scatter.pdf = lambertianPDF(N, scatter.wi);
    scatter.brdf = lambertianBRDF(albedo);
    return scatter;
}

#endif