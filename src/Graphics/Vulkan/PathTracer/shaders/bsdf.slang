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

public float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

public float fresnelDielectric(float cosThetaI, float eta) {
    float sin2I = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sin2T = eta * eta * sin2I;
    if (sin2T >= 1.0) return 1.0;
    float cosThetaT = sqrt(1.0 - sin2T);
    float rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5 * (rs * rs + rp * rp);
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
    public float  alphaX;
    public float  alphaY;

    float distribution(float3 H) {
        float d = H.x * H.x / (alphaX * alphaX) + H.y * H.y / (alphaY * alphaY) + H.z * H.z;
        return 1.0 / max(M_PI * alphaX * alphaY * d * d, 1e-7);
    }

    float lambda(float3 w) {
        float t = (alphaX * alphaX * w.x * w.x + alphaY * alphaY * w.y * w.y) / max(w.z * w.z, 1e-7);
        return (sqrt(1.0 + t) - 1.0) * 0.5;
    }

    public float3 f(float3 wo, float3 wi) {
        if (wo.z <= 0.0 || wi.z <= 0.0) return float3(0.0);
        float3 H = normalize(wo + wi);
        float3 F = fresnelSchlick(saturate(dot(wo, H)), F0);
        float  D = distribution(H);
        float  G = 1.0 / (1.0 + lambda(wo) + lambda(wi));
        float3 spec = D * G / (4.0 * wo.z * wi.z) * F;
        return spec * multiScatterComp(F0, wo.z, sqrt(alphaX * alphaY));
    }

    float3 sampleVNDF(float3 wo, inout uint seed) {
        float3 Vh = normalize(float3(alphaX * wo.x, alphaY * wo.y, wo.z));
        float  lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
        float3 T2 = cross(Vh, T1);
        float u1  = rand01(seed);
        float u2  = rand01(seed);
        float r   = sqrt(u1);
        float phi = 2.0 * M_PI * u2;
        float t1  = r * cos(phi);
        float t2  = r * sin(phi);
        float sc  = 0.5 * (1.0 + Vh.z);
        t2 = (1.0 - sc) * sqrt(max(0.0, 1.0 - t1 * t1)) + sc * t2;
        float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
        return normalize(float3(alphaX * Nh.x, alphaY * Nh.y, max(1e-6, Nh.z)));
    }

    public float pdf(float3 wo, float3 wi) {
        if (wo.z <= 0.0 || wi.z <= 0.0) return 0.0;
        float3 H  = normalize(wo + wi);
        float  G1 = 1.0 / (1.0 + lambda(wo));
        return distribution(H) * G1 / (4.0 * wo.z);
    }

    public Scatter sampleF(float3 wo, inout uint seed) {
        Scatter s;
        s.wi  = float3(0.0, 0.0, 1.0);
        s.f   = float3(0.0);
        s.pdf = 0.0;
        if (wo.z <= 0.0) return s;
        float3 H  = sampleVNDF(wo, seed);
        float3 wi = reflect(-wo, H);
        if (wi.z <= 0.0) return s;
        s.wi  = wi;
        s.f   = f(wo, wi);
        s.pdf = pdf(wo, wi);
        return s;
    }
};

public struct Dielectric {
    public float eta;

    public Scatter sampleF(float3 wo, float3 N, inout uint seed) {
        float cosI = saturate(dot(wo, N));
        float F    = fresnelDielectric(cosI, eta);
        Scatter s;
        s.pdf = 1.0;
        if (rand01(seed) < F) {
            s.wi = reflect(-wo, N);
            s.f  = float3(1.0);
        } else {
            float3 t = refract(-wo, N, eta);
            if (dot(t, t) < 1e-8) {
                s.wi = reflect(-wo, N);
                s.f  = float3(1.0);
            } else {
                s.wi = normalize(t);
                s.f  = float3(eta * eta);
            }
        }
        return s;
    }
};

public Dielectric makeDielectric(float eta) {
    Dielectric d;
    d.eta = eta;
    return d;
}

public struct RoughDielectric {
    public float eta;    // etaT / etaI (ior entering, 1/ior exiting)
    public float alpha;

    float D(float3 wm) {
        float a2 = alpha * alpha;
        float t  = (wm.x * wm.x + wm.y * wm.y) / a2 + wm.z * wm.z;
        return 1.0 / max(M_PI * a2 * t * t, 1e-9);
    }
    float lambda(float3 w) {
        float t = alpha * alpha * (w.x * w.x + w.y * w.y) / max(w.z * w.z, 1e-9);
        return (sqrt(1.0 + t) - 1.0) * 0.5;
    }
    float G1(float3 w) { return 1.0 / (1.0 + lambda(w)); }
    float G(float3 wo, float3 wi) { return 1.0 / (1.0 + lambda(wo) + lambda(wi)); }

    float fr(float cosI) {
        float e  = eta;
        float ci = clamp(cosI, -1.0, 1.0);
        if (ci < 0.0) { e = 1.0 / e; ci = -ci; }
        float s2t = (1.0 - ci * ci) / (e * e);
        if (s2t >= 1.0) return 1.0;
        float ct = sqrt(1.0 - s2t);
        float rp = (e * ci - ct) / (e * ci + ct);
        float rs = (ci - e * ct) / (ci + e * ct);
        return 0.5 * (rp * rp + rs * rs);
    }

    float3 sampleWm(float3 wo, inout uint seed) {
        float3 Vh = normalize(float3(alpha * wo.x, alpha * wo.y, wo.z));
        float  lensq = Vh.x * Vh.x + Vh.y * Vh.y;
        float3 T1 = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
        float3 T2 = cross(Vh, T1);
        float u1 = rand01(seed);
        float u2 = rand01(seed);
        float r  = sqrt(u1);
        float phi = 2.0 * M_PI * u2;
        float t1 = r * cos(phi);
        float t2 = r * sin(phi);
        float sc = 0.5 * (1.0 + Vh.z);
        t2 = (1.0 - sc) * sqrt(max(0.0, 1.0 - t1 * t1)) + sc * t2;
        float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
        return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(1e-6, Nh.z)));
    }

    public float3 f(float3 wo, float3 wi) {
        float coso = wo.z, cosi = wi.z;
        if (coso == 0.0 || cosi == 0.0) return float3(0.0);
        bool  refl = cosi * coso > 0.0;
        float etap = refl ? 1.0 : (coso > 0.0 ? eta : (1.0 / eta));
        float3 wm = wi * etap + wo;
        if (dot(wm, wm) == 0.0) return float3(0.0);
        wm = normalize(wm);
        if (wm.z < 0.0) wm = -wm;
        if (dot(wm, wi) * cosi < 0.0 || dot(wm, wo) * coso < 0.0) return float3(0.0);
        float F = fr(dot(wo, wm));
        if (refl) {
            return float3(D(wm) * G(wo, wi) * F / abs(4.0 * cosi * coso));
        }
        float d     = dot(wi, wm) + dot(wo, wm) / etap;
        float denom = d * d * cosi * coso;
        float ft    = D(wm) * (1.0 - F) * G(wo, wi) * abs(dot(wi, wm) * dot(wo, wm) / denom);
        return float3(ft / (etap * etap));
    }

    public float pdf(float3 wo, float3 wi) {
        float coso = wo.z, cosi = wi.z;
        if (coso == 0.0 || cosi == 0.0) return 0.0;
        bool  refl = cosi * coso > 0.0;
        float etap = refl ? 1.0 : (coso > 0.0 ? eta : (1.0 / eta));
        float3 wm = wi * etap + wo;
        if (dot(wm, wm) == 0.0) return 0.0;
        wm = normalize(wm);
        if (wm.z < 0.0) wm = -wm;
        if (dot(wm, wi) * cosi < 0.0 || dot(wm, wo) * coso < 0.0) return 0.0;
        float R = fr(dot(wo, wm));
        float T = 1.0 - R;
        float vpdf = G1(wo) * D(wm) * abs(dot(wo, wm)) / max(abs(coso), 1e-9);
        if (refl) {
            return vpdf / (4.0 * abs(dot(wo, wm))) * R;
        }
        float d   = dot(wi, wm) + dot(wo, wm) / etap;
        float dwm = abs(dot(wi, wm)) / (d * d);
        return vpdf * dwm * T;
    }

    public Scatter sampleF(float3 wo, inout uint seed) {
        Scatter s;
        s.wi  = float3(0.0, 0.0, 1.0);
        s.f   = float3(0.0);
        s.pdf = 0.0;
        if (wo.z == 0.0) return s;
        float3 wm = sampleWm(wo, seed);
        float  R  = fr(dot(wo, wm));
        if (rand01(seed) < R) {
            float3 wi = reflect(-wo, wm);
            if (wi.z * wo.z <= 0.0) return s;
            s.wi  = wi;
            s.f   = f(wo, wi);
            s.pdf = pdf(wo, wi);
        } else {
            float  e  = eta;
            float3 n  = wm;
            float  ci = dot(n, wo);
            if (ci < 0.0) { e = 1.0 / e; ci = -ci; n = -n; }
            float s2t = (1.0 - ci * ci) / (e * e);
            if (s2t >= 1.0) return s;
            float  ct = sqrt(1.0 - s2t);
            float3 wi = -wo / e + (ci / e - ct) * n;
            if (wi.z * wo.z >= 0.0) return s;
            s.wi  = wi;
            s.f   = f(wo, wi);
            s.pdf = pdf(wo, wi);
        }
        return s;
    }
};

public RoughDielectric makeRoughDielectric(float eta, float roughness) {
    RoughDielectric d;
    d.eta   = eta;
    d.alpha = max(roughness * roughness, 1e-4);
    return d;
}

public struct BSDF {
    public float3     ns;
    public float3     ss;
    public float3     ts;
    public Lambertian diffuse;
    public GGX        specular;
    public GGX        coat;
    public float      clearcoat;
    public float      pSpec;
    public float      pCoat;

    public float3 toLocal(float3 v) { return float3(dot(v, ss), dot(v, ts), dot(v, ns)); }
    public float3 toWorld(float3 v) { return v.x * ss + v.y * ts + v.z * ns; }

    float3 evalLocal(float3 wo, float3 wi) {
        if (wo.z <= 0.0 || wi.z <= 0.0) return float3(0.0);
        float3 H = normalize(wo + wi);
        float3 Fbase = fresnelSchlick(saturate(dot(wo, H)), specular.F0);
        float3 base  = diffuse.f(wo, wi) * (1.0 - Fbase) + specular.f(wo, wi);
        float  coatFr   = clearcoat * (0.04 + 0.96 * pow(saturate(1.0 - dot(wo, H)), 5.0));
        float3 coatSpec = clearcoat * coat.f(wo, wi);
        return base * (1.0 - coatFr) + coatSpec;
    }

    float pdfLocal(float3 wo, float3 wi) {
        float basePdf = lerp(diffuse.pdf(wo, wi), specular.pdf(wo, wi), pSpec);
        return lerp(basePdf, coat.pdf(wo, wi), pCoat);
    }

    public float3 f(float3 woWorld, float3 wiWorld) {
        return evalLocal(toLocal(woWorld), toLocal(wiWorld));
    }

    public float pdf(float3 woWorld, float3 wiWorld) {
        return pdfLocal(toLocal(woWorld), toLocal(wiWorld));
    }

    public Scatter sampleF(float3 woWorld, inout uint seed) {
        float3 wo = toLocal(woWorld);
        Scatter s;
        if (rand01(seed) < pCoat) {
            s = coat.sampleF(wo, seed);
        } else if (rand01(seed) < pSpec) {
            s = specular.sampleF(wo, seed);
        } else {
            s = diffuse.sampleF(wo, seed);
        }
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

public BSDF makeBSDF(float3 N, float4 tangent, float3 albedo, float metalness, float roughness,
                     float anisotropy, float clearcoat, float clearcoatRoughness) {
    float3 ss, ts;
    if (dot(tangent.xyz, tangent.xyz) > 1e-8) {
        ss = normalize(tangent.xyz - N * dot(tangent.xyz, N));
        ts = cross(N, ss);
    } else {
        createONB(N, ss, ts);
    }

    BSDF b;
    b.ns = N;
    b.ss = ss;
    b.ts = ts;
    b.diffuse.albedo = albedo * (1.0 - metalness);
    b.specular.F0    = lerp(float3(0.04), albedo, metalness);
    float alpha  = max(roughness * roughness, 1e-3);
    float aspect = sqrt(1.0 - 0.9 * saturate(anisotropy));
    b.specular.alphaX = max(alpha / aspect, 1e-3);
    b.specular.alphaY = max(alpha * aspect, 1e-3);

    b.clearcoat   = saturate(clearcoat);
    b.coat.F0     = float3(0.04);
    float ccAlpha = max(clearcoatRoughness * clearcoatRoughness, 1e-3);
    b.coat.alphaX = ccAlpha;
    b.coat.alphaY = ccAlpha;

    float lumSpec = luminance(b.specular.F0);
    float lumDiff = luminance(b.diffuse.albedo);
    b.pSpec = clamp(lumSpec / (lumSpec + lumDiff + 1e-4), 0.1, 0.9);
    b.pCoat = clamp(b.clearcoat * 0.5, 0.0, 0.5);
    return b;
}
