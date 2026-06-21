module bsdf;

import common;

public struct Scatter {
    public float3 wi;
    public float3 brdf;
    public float  pdf;
};

public struct Surface {
    public float3 N;
    public float3 wo;
    public float3 diffColor;
    public float3 F0;
    public float  alpha;
    public float  pSpec;
    public float  ccWeight;
    public float  ccAlpha;
    public float  pClearcoat;
};

public void createONB(float3 N, out float3 tangent, out float3 bitangent) {
    float s = copySign(1.0, N.z);
    float a = -1.0 / (s + N.z);
    float b = N.x * N.y * a;
    tangent   = float3(1.0 + s * N.x * N.x * a, s * b, -s * N.x);
    bitangent = float3(b, s + N.y * N.y * a, -N.y);
}

public float3 cosineSample(inout uint seed) {
    float u = rand01(seed);
    float v = rand01(seed);
    float r = sqrt(u);
    float azimuth = v * 2.0 * M_PI;
    return float3(r * cos(azimuth), r * sin(azimuth), sqrt(1.0 - u));
}

public Surface makeSurface(float3 baseColor, float metalness, float roughness, float ior, float specular,
                           float clearcoat, float clearcoatRoughness, float3 N, float3 wo) {
    Surface s;
    s.N         = N;
    s.wo        = wo;
    float  r            = (ior - 1.0) / (ior + 1.0);
    float  dielectricF0 = min(r * r * specular, 1.0);
    s.diffColor = baseColor * (1.0 - metalness);
    s.F0        = lerp(float3(dielectricF0), baseColor, metalness);
    s.alpha     = max(roughness * roughness, 1e-3);
    float fl    = luminance(s.F0);
    float dl    = luminance(s.diffColor);
    s.pSpec     = clamp(fl / (fl + dl + 1e-4), 0.1, 0.9);
    s.ccWeight  = clearcoat;
    s.ccAlpha   = max(clearcoatRoughness * clearcoatRoughness, 1e-3);
    s.pClearcoat = clamp(clearcoat * 0.25, 0.0, 0.25);
    return s;
}

float D_GGX(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(M_PI * d * d, 1e-7);
}

float V_SmithGGXCorrelated(float NdotV, float NdotL, float alpha) {
    float a2      = alpha * alpha;
    float lambdaV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float lambdaL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(lambdaV + lambdaL, 1e-7);
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0);
}

float fresnelDielectric(float cosThetaI, float eta) {
    float sin2I = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sin2T = eta * eta * sin2I;
    if (sin2T >= 1.0) return 1.0;
    float cosThetaT = sqrt(1.0 - sin2T);
    float rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    return 0.5 * (rs * rs + rp * rp);
}

public Scatter sampleDielectric(inout uint seed, float3 N, float3 wo, float ior, bool frontFace, float3 tint) {
    Scatter scatter;
    float  eta  = frontFace ? (1.0 / ior) : ior;
    float  cosI = saturate(dot(N, wo));
    float  F    = fresnelDielectric(cosI, eta);

    float3 incident   = -wo;
    float3 refractDir = refract(incident, N, eta);
    bool   tir        = dot(refractDir, refractDir) < 1e-8;

    if (tir || rand01(seed) < F) {
        scatter.wi   = normalize(reflect(incident, N));
        scatter.brdf = float3(1.0);
    } else {
        scatter.wi   = normalize(refractDir);
        scatter.brdf = tint;
    }
    scatter.pdf = -1.0;
    return scatter;
}

float3 sampleGGX(inout uint seed, float3 wo, float3 tangent, float3 bitangent, float3 N, float alpha) {
    float u    = rand01(seed);
    float v    = rand01(seed);
    float phi  = 2.0 * M_PI * u;
    float cosT = sqrt((1.0 - v) / (1.0 + (alpha * alpha - 1.0) * v));
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    float3 Hl  = float3(sinT * cos(phi), sinT * sin(phi), cosT);
    float3 H   = Hl.x * tangent + Hl.y * bitangent + Hl.z * N;
    return reflect(-wo, H);
}

public void evalBSDF(Surface s, float3 wi, out float3 brdf, out float pdf) {
    brdf = float3(0.0);
    pdf  = 0.0;

    float NdotL = dot(s.N, wi);
    float NdotV = dot(s.N, s.wo);
    if (NdotL <= 0.0 || NdotV <= 0.0) return;

    float3 H     = normalize(s.wo + wi);
    float  NdotH = saturate(dot(s.N, H));
    float  VdotH = saturate(dot(s.wo, H));

    float3 F   = fresnelSchlick(VdotH, s.F0);
    float  D   = D_GGX(NdotH, s.alpha);
    float  Vis = V_SmithGGXCorrelated(NdotV, NdotL, s.alpha);

    float3 specular = D * Vis * F;
    float3 diffuse  = s.diffColor * M_1_PI * (1.0 - F);
    float3 baseBrdf = diffuse + specular;

    float diffPdf = NdotL * M_1_PI;
    float specPdf = D * NdotH / max(4.0 * VdotH, 1e-7);
    float basePdf = lerp(diffPdf, specPdf, s.pSpec);

    float  Fcc    = 0.04 + 0.96 * pow(saturate(1.0 - VdotH), 5.0);
    float  Dcc    = D_GGX(NdotH, s.ccAlpha);
    float  Viscc  = V_SmithGGXCorrelated(NdotV, NdotL, s.ccAlpha);
    float  ccSpec = Dcc * Viscc * Fcc * s.ccWeight;
    float  ccPdf  = Dcc * NdotH / max(4.0 * VdotH, 1e-7);

    brdf = baseBrdf * (1.0 - s.ccWeight * Fcc) + ccSpec;
    pdf  = lerp(basePdf, ccPdf, s.pClearcoat);
}

public Scatter sampleBSDF(inout uint seed, Surface s) {
    Scatter scatter;
    float3 tangent, bitangent;
    createONB(s.N, tangent, bitangent);

    float3 wi;
    if (rand01(seed) < s.pClearcoat) {
        wi = sampleGGX(seed, s.wo, tangent, bitangent, s.N, s.ccAlpha);
    } else if (rand01(seed) < s.pSpec) {
        wi = sampleGGX(seed, s.wo, tangent, bitangent, s.N, s.alpha);
    } else {
        float3 local = cosineSample(seed);
        wi = local.x * tangent + local.y * bitangent + local.z * s.N;
    }

    scatter.wi = normalize(wi);
    evalBSDF(s, scatter.wi, scatter.brdf, scatter.pdf);
    return scatter;
}
