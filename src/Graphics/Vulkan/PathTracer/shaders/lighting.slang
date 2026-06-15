module lighting;

import common;
import types;

public struct LightSample {
    public float3 P;
    public float3 N;
    public float3 emission;
    public float  pdf;
};

public float emitterMisWeight(MeshInfo mesh, float objArea, float3 we1, float3 we2,
                              float3 N, float3 rayDir, float dist, uint emissiveCount, float prevBrdfPdf) {
    uint  triCount  = mesh.indexCount / 3;
    float worldArea = 0.5 * length(cross(we1, we2));
    float cosLight  = abs(dot(N, rayDir));
    if (worldArea <= PDF_EPS || cosLight <= PDF_EPS || emissiveCount == 0u)
        return 1.0;
    bool  useAlias   = mesh.aliasAddress != 0 && mesh.area > 0.0;
    float selectProb = useAlias ? (objArea / mesh.area) : (1.0 / float(triCount));
    float areaPdf    = selectProb / worldArea / float(emissiveCount);
    float pdfLight   = areaPdf * dist * dist / cosLight;
    return powerHeuristic(prevBrdfPdf, pdfLight);
}

public LightSample sampleEmissiveMesh(inout uint seed, InstanceData instance, MeshInfo mesh, MaterialData material) {
    uint triCount = mesh.indexCount / 3;

    bool useAlias = mesh.aliasAddress != 0 && mesh.area > 0.0;
    uint triIndex = min(uint(rand01(seed) * float(triCount)), triCount - 1u);
    if (useAlias) {
        AliasEntry* aliasTable = (AliasEntry*)mesh.aliasAddress;
        AliasEntry  entry      = aliasTable[triIndex];
        triIndex = (rand01(seed) < entry.prob) ? triIndex : entry.alias;
    }

    Vertex v0 = mesh.vertices[mesh.indices[triIndex * 3 + 0]];
    Vertex v1 = mesh.vertices[mesh.indices[triIndex * 3 + 1]];
    Vertex v2 = mesh.vertices[mesh.indices[triIndex * 3 + 2]];

    float r1 = sqrt(rand01(seed));
    float r2 = rand01(seed);
    float bu = 1.0 - r1;
    float bv = r1 * (1.0 - r2);
    float bw = r1 * r2;

    float4x4 xform = instance.model;
    float3 lP = bu * v0.pos + bv * v1.pos + bw * v2.pos;
    float3 lN = normalize(bu * v0.n + bv * v1.n + bw * v2.n);

    lP = mul(xform, float4(lP, 1.0)).xyz;
    float3x3 nm = transpose(inverse3x3((float3x3)xform));
    lN = normalize(mul(nm, lN));

    float  objArea   = 0.5 * length(cross(v1.pos - v0.pos, v2.pos - v0.pos));
    float3 e1        = mul(xform, float4(v1.pos - v0.pos, 0.0)).xyz;
    float3 e2        = mul(xform, float4(v2.pos - v0.pos, 0.0)).xyz;
    float  worldArea = 0.5 * length(cross(e1, e2));

    LightSample ls;
    ls.P        = lP;
    ls.N        = lN;
    ls.emission = material.emissionColor * material.luminance;
    float selectProb = useAlias ? (objArea / mesh.area) : (1.0 / float(triCount));
    ls.pdf      = (worldArea > PDF_EPS) ? selectProb / worldArea : 0.0;
    return ls;
}
