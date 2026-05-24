#ifndef TRIANGLE_GLSL
#define TRIANGLE_GLSL

#include "utils.glsl"

struct PrecomputedTriangle {
    vec4 p0, e1, e2, N;
};

struct TriangleExtra {
    vec4 n0, n1, n2;
    vec2 uv0, uv1, uv2;
    float padding[2];
};

struct TriHit {
    float t, u, v;
};

TriHit intersect(PrecomputedTriangle tri, Ray ray, float tmin_eps) {
    vec3 p0 = tri.p0.xyz;
    vec3 e1 = tri.e1.xyz;
    vec3 e2 = tri.e2.xyz;
    vec3 N  = tri.N.xyz;

    vec3 c = p0 - ray.pos;
    vec3 r = cross(ray.dir, c);
    float inv_det = 1.0 / dot(N, ray.dir);

    float u = dot(r, e2) * inv_det;
    float v = dot(r, e1) * inv_det;
    float w = 1.0 - u - v;

    if (u >= eps && v >= eps && w >= eps) {
        float t = dot(N, c) * inv_det;
        if (t >= ray.tmin && t <= ray.tmax)
        return TriHit(t, u, v);
    }

    return TriHit(-1, 0, 0);
}

vec2 getUV(TriangleExtra triExtra, float u, float v) {
    const float w = 1.0 - u - v;
    return u * triExtra.uv1 +
           v * triExtra.uv2 +
           w * triExtra.uv0;
}

#endif