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

float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(M_PI * d * d, 1e-7);
}

float V_SmithGGX(float NdotV, float NdotL, float alpha) {
    float a2      = alpha * alpha;
    float lambdaV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float lambdaL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(lambdaV + lambdaL, 1e-7);
}

public float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float3 sampleGGXHalf(float alpha, inout uint seed) {
    float u1   = rand01(seed);
    float u2   = rand01(seed);
    float phi  = 2.0 * M_PI * u1;
    float cosT = sqrt((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2));
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    return float3(sinT * cos(phi), sinT * sin(phi), cosT);
}

float3 multiScatterComp(float3 F0, float NdotV, float alpha) {
    float  roughness = sqrt(alpha);
    float4 c0   = float4(-1.0, -0.0275, -0.572, 0.022);
    float4 c1   = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r    = roughness * c0 + c1;
    float  a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    float2 AB   = float2(-1.04, 1.04) * a004 + r.zw;
    float  Ess  = AB.x + AB.y;
    return 1.0 + F0 * (1.0 - Ess) / max(Ess, 1e-3);
}

public struct Scatter {
    public float3 wi;
    public float3 f;
    public float  pdf;
};

public interface BRDF {
    float3 f(float3 wo, float3 wi);
    float  pdf(float3 wo, float3 wi);
    Scatter sampleF(float3 wo, inout uint seed);
};

public struct Lambertian : BRDF {
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

public struct GGX : BRDF {
    public float3 F0;
    public float  alpha;

    public float3 f(float3 wo, float3 wi) {
        if (wo.z <= 0.0 || wi.z <= 0.0) return float3(0.0);
        float3 H     = normalize(wo + wi);
        float  NdotH = saturate(H.z);
        float  VdotH = saturate(dot(wo, H));
        float3 F = fresnelSchlick(VdotH, F0);
        float  D = D_GGX(NdotH, alpha);
        float  V = V_SmithGGX(wo.z, wi.z, alpha);
        return D * V * F * multiScatterComp(F0, wo.z, alpha);
    }

    public float pdf(float3 wo, float3 wi) {
        if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;
        float3 H     = normalize(wo + wi);
        float  NdotH = saturate(H.z);
        float  VdotH = max(dot(wo, H), 1e-7);
        return D_GGX(NdotH, alpha) * NdotH / (4.0 * VdotH);
    }

    public Scatter sampleF(float3 wo, inout uint seed) {
        Scatter s;
        s.wi  = float3(0.0, 0.0, 1.0);
        s.f   = float3(0.0);
        s.pdf = 0.0;
        if (wo.z <= 0.0) return s;
        float3 H  = sampleGGXHalf(alpha, seed);
        float3 wi = reflect(-wo, H);
        if (wi.z <= 0.0) return s;
        s.wi  = wi;
        s.f   = f(wo, wi);
        s.pdf = pdf(wo, wi);
        return s;
    }
};

public struct BSDF {
    public float3     ns;
    public float3     ss;
    public float3     ts;
    public Lambertian diffuse;
    public GGX        specular;
    public float      pSpec;

    public float3 toLocal(float3 v) { return float3(dot(v, ss), dot(v, ts), dot(v, ns)); }
    public float3 toWorld(float3 v) { return v.x * ss + v.y * ts + v.z * ns; }

    float3 evalLocal(float3 wo, float3 wi) {
        if (wo.z <= 0.0 || wi.z <= 0.0) return float3(0.0);
        float3 H = normalize(wo + wi);
        float3 F = fresnelSchlick(saturate(dot(wo, H)), specular.F0);
        return diffuse.f(wo, wi) * (1.0 - F) + specular.f(wo, wi);
    }

    float pdfLocal(float3 wo, float3 wi) {
        return lerp(diffuse.pdf(wo, wi), specular.pdf(wo, wi), pSpec);
    }

    public float3 f(float3 woWorld, float3 wiWorld) {
        return evalLocal(toLocal(woWorld), toLocal(wiWorld));
    }

    public float pdf(float3 woWorld, float3 wiWorld) {
        return pdfLocal(toLocal(woWorld), toLocal(wiWorld));
    }

    public Scatter sampleF(float3 woWorld, inout uint seed) {
        float3 wo = toLocal(woWorld);
        Scatter s = (rand01(seed) < pSpec) ? specular.sampleF(wo, seed)
                                           : diffuse.sampleF(wo, seed);
        if (s.pdf <= 0.0) {
            s.f   = float3(0.0);
            s.pdf = 0.0;
            return s;
        }
        s.f   = evalLocal(wo, s.wi);
        s.pdf = pdfLocal(wo, s.wi);
        s.wi  = toWorld(s.wi);
        return s;
    }
};

public BSDF makeBSDF(float3 N, float3 albedo, float metalness, float roughness) {
    float3 ss, ts;
    createONB(N, ss, ts);

    BSDF b;
    b.ns = N;
    b.ss = ss;
    b.ts = ts;
    b.diffuse.albedo = albedo * (1.0 - metalness);
    b.specular.F0    = lerp(float3(0.04), albedo, metalness);
    b.specular.alpha = max(roughness * roughness, 1e-3);
    float lumSpec = luminance(b.specular.F0);
    float lumDiff = luminance(b.diffuse.albedo);
    b.pSpec = clamp(lumSpec / (lumSpec + lumDiff + 1e-4), 0.1, 0.9);
    return b;
}
