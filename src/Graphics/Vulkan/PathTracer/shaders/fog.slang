module fog;

import common;
import scene;
import bsdf;

public bool fogEnabled() {
    return fogData.density > 0.0;
}

float fogScaleAt(float y) {
    if (fogData.height <= 0.0) return 1.0;
    return exp(clamp(-(y - fogData.base) / fogData.height, -80.0, 80.0));
}

public float fogOpticalDepth(float3 origin, float3 dir, float dist) {
    if (!fogEnabled() || dist <= 0.0) return 0.0;
    float a = fogData.density * fogScaleAt(origin.y);
    if (a <= 0.0) return 0.0;
    if (fogData.height <= 0.0) return a * dist;

    float b = dir.y / fogData.height;
    if (abs(b) < 1e-6) return a * dist;
    float e = exp(clamp(-b * dist, -80.0, 80.0));
    return max(a * (1.0 - e) / b, 0.0);
}

public float fogTransmittance(float3 origin, float3 dir, float dist) {
    return exp(-fogOpticalDepth(origin, dir, dist));
}

public bool fogSampleDistance(float3 origin, float3 dir, float tMax, inout uint seed, out float t) {
    t = 0.0;
    if (!fogEnabled()) return false;
    float a = fogData.density * fogScaleAt(origin.y);
    if (a <= 0.0) return false;

    float target = -log(max(1.0 - rand01(seed), 1e-9));
    float b = fogData.height <= 0.0 ? 0.0 : dir.y / fogData.height;
    if (abs(b) < 1e-6) {
        t = target / a;
    } else {
        float inner = 1.0 - target * b / a;
        if (inner <= 0.0) return false;
        t = -log(inner) / b;
    }
    return t > 0.0 && t < tMax;
}

public float hgPhase(float cosTheta, float g) {
    float g2 = g * g;
    float d  = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-6);
    return (1.0 - g2) / (4.0 * M_PI * d * sqrt(d));
}

public float3 hgSample(float3 dir, float g, inout uint seed, out float pdf) {
    float u1 = rand01(seed);
    float u2 = rand01(seed);

    float cosTheta;
    if (abs(g) < 1e-3) {
        cosTheta = 1.0 - 2.0 * u1;
    } else {
        float s  = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        cosTheta = (1.0 + g * g - s * s) / (2.0 * g);
    }
    cosTheta = clamp(cosTheta, -1.0, 1.0);

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi      = 2.0 * M_PI * u2;
    float3 tangent, bitangent;
    createONB(dir, tangent, bitangent);

    pdf = hgPhase(cosTheta, g);
    return sinTheta * cos(phi) * tangent + sinTheta * sin(phi) * bitangent + cosTheta * dir;
}
