module volume;

import common;
import scene;

static const uint MAX_VOLUME_STEPS = 256u;

float volumeHash(float3 p) {
    p = frac(p * 0.3183099 + float3(0.71, 0.113, 0.419));
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float valueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);

    float n000 = volumeHash(i + float3(0.0, 0.0, 0.0));
    float n100 = volumeHash(i + float3(1.0, 0.0, 0.0));
    float n010 = volumeHash(i + float3(0.0, 1.0, 0.0));
    float n110 = volumeHash(i + float3(1.0, 1.0, 0.0));
    float n001 = volumeHash(i + float3(0.0, 0.0, 1.0));
    float n101 = volumeHash(i + float3(1.0, 0.0, 1.0));
    float n011 = volumeHash(i + float3(0.0, 1.0, 1.0));
    float n111 = volumeHash(i + float3(1.0, 1.0, 1.0));

    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);
    return lerp(nxy0, nxy1, u.z);
}

float fbm(float3 p, uint octaves) {
    float sum  = 0.0;
    float amp  = 0.5;
    float norm = 0.0;
    for (uint i = 0u; i < octaves; ++i) {
        sum  += amp * valueNoise(p);
        norm += amp;
        amp  *= 0.5;
        p    *= 2.02;
    }
    return norm > 0.0 ? sum / norm : 0.0;
}

float shapeFalloff(float3 p, uint shape) {
    if (shape == 0u) return saturate(1.0 - length(p));
    float3 a = abs(p);
    return saturate(1.0 - max(a.x, max(a.y, a.z)));
}

float volumeDensityAt(VolumeGPU v, float3 pObj) {
    float d = shapeFalloff(pObj, v.shape);
    if (d <= 0.0) return 0.0;
    if (v.noiseStrength > 0.0)
        d *= lerp(1.0, fbm(pObj * v.noiseScale, v.octaves), v.noiseStrength);
    float c = min(v.coverage, 0.95);
    return v.density * saturate((d - c) / (1.0 - c));
}

bool volumeSegment(VolumeGPU v, float3 o, float3 d, float tMax, out float t0, out float t1) {
    t0 = 0.0;
    t1 = 0.0;
    if (v.shape == 0u) {
        float a = dot(d, d);
        if (a <= 0.0) return false;
        float b    = dot(o, d);
        float c    = dot(o, o) - 1.0;
        float disc = b * b - a * c;
        if (disc <= 0.0) return false;
        float sq = sqrt(disc);
        t0 = (-b - sq) / a;
        t1 = (-b + sq) / a;
    } else {
        float3 inv  = 1.0 / d;
        float3 ta   = (float3(-1.0) - o) * inv;
        float3 tb   = (float3( 1.0) - o) * inv;
        float3 tlo  = min(ta, tb);
        float3 thi  = max(ta, tb);
        t0 = max(tlo.x, max(tlo.y, tlo.z));
        t1 = min(thi.x, min(thi.y, thi.z));
        if (t1 < t0) return false;
    }
    t0 = max(t0, 0.0);
    t1 = min(t1, tMax);
    return t1 > t0;
}

public bool volumeCollision(uint index, float3 origin, float3 dir, float tMax, inout uint seed, out float t) {
    t = 0.0;
    VolumeGPU v = volumes[index];
    if (v.density <= 0.0) return false;

    float3 o = mul(v.invModel, float4(origin, 1.0)).xyz;
    float3 d = mul(v.invModel, float4(dir,    0.0)).xyz;

    float t0, t1;
    if (!volumeSegment(v, o, d, tMax, t0, t1)) return false;

    float tc = t0;
    for (uint step = 0u; step < MAX_VOLUME_STEPS; ++step) {
        tc -= log(max(1.0 - rand01(seed), 1e-9)) / v.density;
        if (tc >= t1) return false;
        if (rand01(seed) * v.density < volumeDensityAt(v, o + d * tc)) {
            t = tc;
            return true;
        }
    }
    return false;
}

public float volumeTransmittance(uint index, float3 origin, float3 dir, float dist, inout uint seed) {
    VolumeGPU v = volumes[index];
    if (v.density <= 0.0) return 1.0;

    float3 o = mul(v.invModel, float4(origin, 1.0)).xyz;
    float3 d = mul(v.invModel, float4(dir,    0.0)).xyz;

    float t0, t1;
    if (!volumeSegment(v, o, d, dist, t0, t1)) return 1.0;

    float tr = 1.0;
    float tc = t0;
    for (uint step = 0u; step < MAX_VOLUME_STEPS; ++step) {
        tc -= log(max(1.0 - rand01(seed), 1e-9)) / v.density;
        if (tc >= t1) break;
        tr *= 1.0 - volumeDensityAt(v, o + d * tc) / v.density;
        if (tr < 1e-4) return 0.0;
    }
    return tr;
}
