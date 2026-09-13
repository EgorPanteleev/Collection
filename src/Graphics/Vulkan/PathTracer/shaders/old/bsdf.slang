module bsdf;                                          // Slang module: microfacet BSDF model (sampling + evaluation)

import common;                                        // RNG, constants, luminance, copySign

public struct Scatter {                               // result of sampling a BSDF direction
    public float3 wi;                                 // sampled incoming (next ray) direction, world space
    public float3 brdf;                               // BSDF value for that direction
    public float  pdf;                                // pdf of the sample (negative = delta/specular, handled specially)
};                                                    // end Scatter

public struct Surface {                               // precomputed shading parameters at a hit, in world space
    public float3 N;                                  // shading normal
    public float3 wo;                                 // outgoing (toward camera) direction
    public float3 T;                                  // tangent (for anisotropy)
    public float3 B;                                  // bitangent (for anisotropy)
    public float3 diffColor;                          // diffuse albedo (base color scaled by non-metalness)
    public float3 F0;                                 // specular reflectance at normal incidence
    public float  alphaX;                             // GGX roughness along tangent
    public float  alphaY;                             // GGX roughness along bitangent
    public float  sheen;                              // sheen strength
    public float  pSpec;                              // probability of choosing the specular lobe when sampling
    public float  ccWeight;                           // clearcoat strength
    public float  ccAlpha;                            // clearcoat GGX roughness
    public float  pClearcoat;                         // probability of choosing the clearcoat lobe when sampling
};                                                    // end Surface

public void createONB(float3 N, out float3 tangent, out float3 bitangent) { // build an orthonormal basis around N (Duff et al. method)
    float s = copySign(1.0, N.z);                     // sign of N.z, avoids singularity
    float a = -1.0 / (s + N.z);                       // helper term
    float b = N.x * N.y * a;                          // helper term
    tangent   = float3(1.0 + s * N.x * N.x * a, s * b, -s * N.x); // first basis vector
    bitangent = float3(b, s + N.y * N.y * a, -N.y);   // second basis vector
}                                                     // end createONB

public float3 cosineSample(inout uint seed) {         // cosine-weighted hemisphere sample (local space, +Z up)
    float u = rand01(seed);                           // random for radius
    float v = rand01(seed);                           // random for azimuth
    float r = sqrt(u);                                // radius = sqrt(u) gives cosine weighting
    float azimuth = v * 2.0 * M_PI;                   // azimuthal angle
    return float3(r * cos(azimuth), r * sin(azimuth), sqrt(1.0 - u)); // point on the hemisphere
}                                                     // end cosineSample

public Surface makeSurface(float3 baseColor, float metalness, float roughness, float ior, float specular, // assemble a Surface from material params
                           float clearcoat, float clearcoatRoughness, float anisotropy, float sheen,
                           float4 tangent, float3 N, float3 wo) {
    Surface s;                                        // output surface
    s.N         = N;                                  // store shading normal
    s.wo        = wo;                                 // store view direction
    float  r            = (ior - 1.0) / (ior + 1.0);  // Fresnel reflectance amplitude from IOR
    float  dielectricF0 = min(r * r * specular, 1.0); // dielectric F0 (reflectance at normal incidence), scaled by specular
    s.diffColor = baseColor * (1.0 - metalness);      // metals have no diffuse term
    s.F0        = lerp(float3(dielectricF0), baseColor, metalness); // metals tint specular with base color

    float alpha  = max(roughness * roughness, 1e-3);  // GGX alpha = roughness^2, clamped away from 0
    float aspect = sqrt(1.0 - 0.9 * saturate(anisotropy)); // anisotropy aspect ratio
    s.alphaX = max(alpha / aspect, 1e-3);             // roughness along tangent
    s.alphaY = max(alpha * aspect, 1e-3);             // roughness along bitangent

    float3 T;                                         // tangent to use
    if (dot(tangent.xyz, tangent.xyz) > 1e-8) {       // valid vertex tangent?
        T = normalize(tangent.xyz - N * dot(tangent.xyz, N)); // orthonormalize it against N
    } else {                                          // otherwise
        float3 bt;                                    // dummy bitangent
        createONB(N, T, bt);                          // fabricate a tangent basis
    }                                                 // end tangent choice
    s.T = T;                                          // store tangent
    s.B = normalize(cross(N, T));                     // bitangent from N x T

    s.sheen     = sheen;                              // store sheen
    float fl    = luminance(s.F0);                    // luminance of specular color
    float dl    = luminance(s.diffColor);             // luminance of diffuse color
    s.pSpec     = clamp(fl / (fl + dl + 1e-4), 0.1, 0.9); // lobe-selection probability weighted by energy, clamped
    s.ccWeight  = clearcoat;                          // clearcoat strength
    s.ccAlpha   = max(clearcoatRoughness * clearcoatRoughness, 1e-3); // clearcoat alpha
    s.pClearcoat = clamp(clearcoat * 0.25, 0.0, 0.25); // clearcoat sampling probability
    return s;                                         // assembled surface
}                                                     // end makeSurface

float D_GGX(float NdotH, float alpha) {               // isotropic GGX/Trowbridge-Reitz normal distribution
    float a2 = alpha * alpha;                         // alpha^2
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;      // denominator base
    return a2 / max(M_PI * d * d, 1e-7);              // D value
}                                                     // end D_GGX

float V_SmithGGXCorrelated(float NdotV, float NdotL, float alpha) { // height-correlated Smith visibility term (G/(4 NdotL NdotV))
    float a2      = alpha * alpha;                    // alpha^2
    float lambdaV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2); // view masking term
    float lambdaL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2); // light masking term
    return 0.5 / max(lambdaV + lambdaL, 1e-7);        // combined visibility
}                                                     // end V_SmithGGXCorrelated

float3 fresnelSchlick(float cosTheta, float3 F0) {    // Schlick Fresnel approximation
    return F0 + (1.0 - F0) * pow(saturate(1.0 - cosTheta), 5.0); // interpolate F0 -> 1 by (1-cos)^5
}                                                     // end fresnelSchlick

float fresnelDielectric(float cosThetaI, float eta) { // exact dielectric Fresnel reflectance (unpolarized)
    float sin2I = max(0.0, 1.0 - cosThetaI * cosThetaI); // sin^2 of incidence
    float sin2T = eta * eta * sin2I;                  // Snell's law -> sin^2 of transmission
    if (sin2T >= 1.0) return 1.0;                     // total internal reflection
    float cosThetaT = sqrt(1.0 - sin2T);              // cos of transmission angle
    float rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT); // s-polarized amplitude
    float rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT); // p-polarized amplitude
    return 0.5 * (rs * rs + rp * rp);                 // average reflectance
}                                                     // end fresnelDielectric

public Scatter sampleDielectric(inout uint seed, float3 N, float3 wo, float ior, bool inside) { // sample reflect/refract at a glass interface
    Scatter scatter;                                  // output
    float  eta  = inside ? ior : (1.0 / ior);         // relative IOR depending on which side we're on
    float  cosI = saturate(dot(N, wo));               // cosine of incidence
    float  F    = fresnelDielectric(cosI, eta);       // reflectance

    float3 incident   = -wo;                          // incoming direction
    float3 refractDir = refract(incident, N, eta);    // refracted direction (zero if TIR)
    bool   tir        = dot(refractDir, refractDir) < 1e-8; // detect total internal reflection

    if (tir || rand01(seed) < F)                      // reflect with probability F (or always on TIR)
        scatter.wi = normalize(reflect(incident, N)); // reflection direction
    else                                              // otherwise
        scatter.wi = normalize(refractDir);           // transmission direction

    scatter.brdf = float3(1.0);                       // perfect specular: value folded into throughput elsewhere
    scatter.pdf  = -1.0;                              // negative pdf flags a delta interaction (skip MIS)
    return scatter;                                   // sampled dielectric scatter
}                                                     // end sampleDielectric

float3 sampleGGX(inout uint seed, float3 wo, float3 tangent, float3 bitangent, float3 N, float alpha) { // sample an isotropic GGX half-vector and reflect
    float u    = rand01(seed);                        // random for azimuth
    float v    = rand01(seed);                        // random for elevation
    float phi  = 2.0 * M_PI * u;                      // azimuth
    float cosT = sqrt((1.0 - v) / (1.0 + (alpha * alpha - 1.0) * v)); // GGX cos(theta) of the half-vector
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));   // sin(theta)
    float3 Hl  = float3(sinT * cos(phi), sinT * sin(phi), cosT); // half-vector in local space
    float3 H   = Hl.x * tangent + Hl.y * bitangent + Hl.z * N; // transform half-vector to world space
    return reflect(-wo, H);                           // reflect view about the half-vector
}                                                     // end sampleGGX

float D_GGX_aniso(float3 H, float3 T, float3 B, float3 N, float ax, float ay) { // anisotropic GGX NDF
    float HdotX = dot(H, T);                          // half-vector tangent component
    float HdotY = dot(H, B);                          // half-vector bitangent component
    float HdotN = dot(H, N);                          // half-vector normal component
    float d = HdotX * HdotX / (ax * ax) + HdotY * HdotY / (ay * ay) + HdotN * HdotN; // quadratic form
    return 1.0 / max(M_PI * ax * ay * d * d, 1e-7);   // anisotropic D value
}                                                     // end D_GGX_aniso

float V_SmithGGX_aniso(float3 V, float3 L, float3 T, float3 B, float NdotV, float NdotL, float ax, float ay) { // anisotropic Smith visibility
    float lambdaV = NdotL * length(float3(ax * dot(V, T), ay * dot(V, B), NdotV)); // view masking
    float lambdaL = NdotV * length(float3(ax * dot(L, T), ay * dot(L, B), NdotL)); // light masking
    return 0.5 / max(lambdaV + lambdaL, 1e-7);        // combined visibility
}                                                     // end V_SmithGGX_aniso

float3 sampleGGXAniso(inout uint seed, Surface s) {   // sample an anisotropic GGX half-vector and reflect
    float u1 = rand01(seed);                          // random 1
    float u2 = rand01(seed);                          // random 2
    float phi = atan2(s.alphaY * sin(2.0 * M_PI * u1), s.alphaX * cos(2.0 * M_PI * u1)); // stretched azimuth
    float cosPhi = cos(phi);                          // cos of azimuth
    float sinPhi = sin(phi);                          // sin of azimuth
    float denom = cosPhi * cosPhi / (s.alphaX * s.alphaX) + sinPhi * sinPhi / (s.alphaY * s.alphaY); // anisotropic scaling
    float tanTheta2 = u2 / (1.0 - u2) / max(denom, 1e-7); // tan^2(theta) of the half-vector
    float cosT = 1.0 / sqrt(1.0 + tanTheta2);         // cos(theta)
    float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));   // sin(theta)
    float3 H = (sinT * cosPhi) * s.T + (sinT * sinPhi) * s.B + cosT * s.N; // half-vector in world space
    return reflect(-s.wo, H);                         // reflect view about the half-vector
}                                                     // end sampleGGXAniso

float3 multiScatterComp(float3 F0, float NdotV, float alpha) { // energy-compensation factor for multiple scattering (Fdez-Aguera)
    float  roughness = sqrt(alpha);                   // roughness from alpha
    float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022); // fit constants
    float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);     // fit constants
    float4 r  = roughness * c0 + c1;                  // interpolate fit
    float  a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y; // analytic DFG approximation term
    float2 AB   = float2(-1.04, 1.04) * a004 + r.zw;  // scale/bias pair
    float  Ess  = AB.x + AB.y;                        // directional albedo estimate
    return 1.0 + F0 * (1.0 - Ess) / max(Ess, 1e-3);   // multiply specular by this to restore lost energy
}                                                     // end multiScatterComp

public void evalBSDF(Surface s, float3 wi, out float3 brdf, out float pdf) { // evaluate BSDF value and pdf for a given incoming direction
    brdf = float3(0.0);                               // default: zero contribution
    pdf  = 0.0;                                        // default: zero pdf

    float NdotL = dot(s.N, wi);                       // cosine to the light/incoming direction
    float NdotV = dot(s.N, s.wo);                     // cosine to the view direction
    if (NdotL <= 0.0 || NdotV <= 0.0) return;         // below the horizon on either side -> no contribution

    float3 H     = normalize(s.wo + wi);              // half-vector
    float  NdotH = saturate(dot(s.N, H));             // cosine normal-half
    float  VdotH = saturate(dot(s.wo, H));            // cosine view-half

    float3 F   = fresnelSchlick(VdotH, s.F0);         // Fresnel term
    float  D   = D_GGX_aniso(H, s.T, s.B, s.N, s.alphaX, s.alphaY); // anisotropic NDF
    float  Vis = V_SmithGGX_aniso(s.wo, wi, s.T, s.B, NdotV, NdotL, s.alphaX, s.alphaY); // visibility term

    float  alphaAvg = sqrt(s.alphaX * s.alphaY);      // average roughness for energy comp
    float3 specular = D * Vis * F * multiScatterComp(s.F0, NdotV, alphaAvg); // specular lobe with energy compensation
    float3 diffuse  = s.diffColor * M_1_PI * (1.0 - F); // Lambertian diffuse, energy left after Fresnel reflection
    float3 sheen    = float3(s.sheen * pow(saturate(1.0 - VdotH), 5.0)); // simple sheen at grazing angles
    float3 baseBrdf = diffuse + specular + sheen;     // combined base-layer BRDF

    float diffPdf = NdotL * M_1_PI;                   // pdf of cosine-sampled diffuse
    float specPdf = D * NdotH / max(4.0 * VdotH, 1e-7); // pdf of GGX-sampled specular (half-vector Jacobian)
    float basePdf = lerp(diffPdf, specPdf, s.pSpec);  // mixture pdf by lobe-selection probability

    float  Fcc    = 0.04 + 0.96 * pow(saturate(1.0 - VdotH), 5.0); // clearcoat Fresnel (fixed F0=0.04)
    float  Dcc    = D_GGX(NdotH, s.ccAlpha);          // clearcoat NDF (isotropic)
    float  Viscc  = V_SmithGGXCorrelated(NdotV, NdotL, s.ccAlpha); // clearcoat visibility
    float  ccSpec = Dcc * Viscc * Fcc * s.ccWeight;   // clearcoat specular lobe
    float  ccPdf  = Dcc * NdotH / max(4.0 * VdotH, 1e-7); // clearcoat pdf

    brdf = baseBrdf * (1.0 - s.ccWeight * Fcc) + ccSpec; // attenuate base by clearcoat reflection, then add clearcoat
    pdf  = lerp(basePdf, ccPdf, s.pClearcoat);        // mixture pdf including clearcoat lobe
}                                                     // end evalBSDF

public Scatter sampleBSDF(inout uint seed, Surface s) { // importance-sample a direction from the combined BSDF
    Scatter scatter;                                  // output
    float3 tangent, bitangent;                        // temp basis for the isotropic clearcoat sample
    createONB(s.N, tangent, bitangent);               // build a basis around N

    float3 wi;                                        // sampled direction
    if (rand01(seed) < s.pClearcoat) {                // choose clearcoat lobe?
        wi = sampleGGX(seed, s.wo, tangent, bitangent, s.N, s.ccAlpha); // sample clearcoat GGX
    } else if (rand01(seed) < s.pSpec) {              // else choose base specular lobe?
        wi = sampleGGXAniso(seed, s);                 // sample anisotropic GGX
    } else {                                          // else diffuse lobe
        float3 local = cosineSample(seed);            // cosine-weighted local sample
        wi = local.x * tangent + local.y * bitangent + local.z * s.N; // to world space
    }                                                 // end lobe selection

    scatter.wi = normalize(wi);                       // normalize sampled direction
    evalBSDF(s, scatter.wi, scatter.brdf, scatter.pdf); // evaluate full BSDF value + pdf for the sample (MIS-consistent)
    return scatter;                                   // sampled scatter
}                                                     // end sampleBSDF
