#ifndef UTILS_GLSL
#define UTILS_GLSL

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#define FLT_MAX 3.402823e38
#define INT_MAX 2147483647
#define UINT_MAX 4294967295u
#define M_PI 3.14159265359
#define M_1_PI 0.3183098861837907

#define ALL_OBJECTS 0xFF
#define T_MIN 0.0
#define T_MAX 1e10

#define ORIGIN_TRESHOLD 0.03125
#define FLOAT_SCALE 1.0 / 65536.0
#define INT_SCALE 256.0
#define SHADOW_EPS 1e-3

#define PDF_EPS 1e-4
#define NEE_CLAMP 50.0

struct Vertex {
    vec3 pos;
    vec2 texCoord;
    vec3 n;
    vec4 tangent;
};
struct MeshInfo {
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint     indexCount;
    float    area;
    uint64_t aliasAddress;
};

struct AliasEntry {
    float prob;
    uint  alias;
};
struct InstanceData {
    mat4 model;
    uint meshIndex;
    uint materialIndex;
};

struct PathPayload {
    vec3  throughput;
    vec3  radiance;
    vec3  origin;
    vec3  direction;
    uint  instanceId;
    uint  seed;
    uint  depth;
    float prevBrdfPdf;
    bool  done;
};

struct MaterialData {
    vec3   baseColor;
    vec3   emissionColor;
    float  luminance;
    uint   baseColorTexIndex;
    uint   normalTexIndex;
};

vec3 offsetRay(const vec3 p, const vec3 n) {
    ivec3 of_i = ivec3(INT_SCALE * n.x, INT_SCALE * n.y, INT_SCALE * n.z);
    vec3 p_i = vec3(intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
    intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
    intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return vec3(abs(p.x) < ORIGIN_TRESHOLD ? p.x + FLOAT_SCALE * n.x : p_i.x,
    abs(p.y) < ORIGIN_TRESHOLD ? p.y + FLOAT_SCALE * n.y : p_i.y,
    abs(p.z) < ORIGIN_TRESHOLD ? p.z + FLOAT_SCALE * n.z : p_i.z);
}

vec3 movedPoint(vec3 p, vec3 gn) {
    return offsetRay(p, gn);
}

uint pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}

float rand01(inout uint seed) {
    seed = pcg(seed);
    return float(seed) / 4294967296.0;
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

float copysign(float x, float y) {
    return abs(x) * (y < 0.0 ? -1.0 : 1.0);
}

float powerHeuristic(float pdfA, float pdfB) {
    float a2 = pdfA * pdfA;
    float b2 = pdfB * pdfB;
    float denom = a2 + b2;
    return denom > 0.0 ? a2 / denom : 0.0;
}

#endif