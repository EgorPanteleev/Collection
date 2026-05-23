#ifndef UTILS_GLSL
#define UTILS_GLSL

#define FLT_MAX 3.402823e38
#define INT_MAX 2147483647
#define UINT_MAX 4294967295u
#define M_PI 3.14159265359
#define M_1_PI 0.3183098861837907

struct BBox {
    vec4 min, max;
};

struct Ray {
    vec3 pos, dir, invDir;
    float tmin, tmax;
};

Ray makeRay(vec3 pos, vec3 dir, float tmin, float tmax) {
    Ray ray;
    ray.pos = pos;
    ray.dir = dir;
    ray.tmin = tmin;
    ray.tmax = tmax;
    ray.invDir = 1.0 / dir;
    return ray;
}

float intersect(BBox bbox, Ray ray) {
    vec3 tmin = (bbox.min.xyz - ray.pos) * ray.invDir;
    vec3 tmax = (bbox.max.xyz - ray.pos) * ray.invDir;

    vec3 t1 = min(tmin, tmax);
    vec3 t2 = max(tmin, tmax);

    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar  = min(min(t2.x, t2.y), t2.z);

    if (tFar >= max(tNear, ray.tmin) && tNear < ray.tmax)
    return tNear;
    return FLT_MAX;
}

uint pcg(uint v){
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28) + 4)) ^ state) * 277803737u;
    return (word >> 22) ^ word;
}

float rand01(inout uint seed){
    seed = pcg(seed);
    return float(seed) / 4294967296.0;
}

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

#endif
