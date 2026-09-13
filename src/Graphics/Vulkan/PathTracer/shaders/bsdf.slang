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

public struct Scatter {
    public float3 wi;
    public float3 f;
    public float  pdf;
};

public interface BxDF {
    float3 f(float3 wo, float3 wi);
    float  pdf(float3 wo, float3 wi);
    Scatter sampleF(float3 wo, inout uint seed);
};

public struct Lambertian : BxDF {
    public float3 albedo;

    public float3 f(float3 wo, float3 wi) {
        if (wo.z * wi.z <= 0.0) return float3(0.0);
        return albedo * M_1_PI;
    }

    public float pdf(float3 wo, float3 wi) {
        if (wo.z * wi.z <= 0.0) return 0.0;
        return abs(wi.z) * M_1_PI;
    }

    public Scatter sampleF(float3 wo, inout uint seed) {
        float3 wi = cosineSampleLocal(seed);
        if (wo.z < 0.0) wi.z = -wi.z;
        Scatter s;
        s.wi  = wi;
        s.f   = f(wo, wi);
        s.pdf = pdf(wo, wi);
        return s;
    }
};

public struct BSDF<TBxDF : BxDF> {
    public float3 ns;
    public float3 ss;
    public float3 ts;
    public TBxDF  bxdf;

    public float3 toLocal(float3 v) { return float3(dot(v, ss), dot(v, ts), dot(v, ns)); }
    public float3 toWorld(float3 v) { return v.x * ss + v.y * ts + v.z * ns; }

    public float3 f(float3 woWorld, float3 wiWorld) {
        return bxdf.f(toLocal(woWorld), toLocal(wiWorld));
    }

    public float pdf(float3 woWorld, float3 wiWorld) {
        return bxdf.pdf(toLocal(woWorld), toLocal(wiWorld));
    }

    public Scatter sampleF(float3 woWorld, inout uint seed) {
        Scatter s = bxdf.sampleF(toLocal(woWorld), seed);
        s.wi = toWorld(s.wi);
        return s;
    }
};

public BSDF<Lambertian> makeLambertian(float3 N, float3 albedo) {
    float3 ss, ts;
    createONB(N, ss, ts);
    Lambertian lam;
    lam.albedo = albedo;
    BSDF<Lambertian> b;
    b.ns   = N;
    b.ss   = ss;
    b.ts   = ts;
    b.bxdf = lam;
    return b;
}
