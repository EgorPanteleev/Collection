module lighting;                                      // Slang module: emissive-mesh light sampling + MIS weights

import common;                                        // pdf/MIS helpers, inverse3x3, constants
import types;                                         // Vertex/MeshGPU/InstanceGPU/AliasEntry

public struct LightSample {                           // a point sampled on an emissive surface
    public float3 P;                                  // world-space position of the sample
    public float3 N;                                  // world-space normal at the sample
    public float2 uv;                                 // texture coordinates at the sample
    public float3 emission;                           // emitted radiance (filled in by the caller, which owns textures)
    public float  pdf;                                // solid-angle-ready area pdf of the sample
};                                                    // end LightSample

public float emitterMisWeight(MeshGPU mesh, float objArea, float3 we1, float3 we2, // MIS weight when a BSDF ray randomly hits an emitter
                              float3 N, float3 rayDir, float dist, uint emissiveCount, float prevBrdfPdf) {
    uint  triCount  = mesh.indexCount / 3;            // number of triangles in the emitter mesh
    float worldArea = 0.5 * length(cross(we1, we2));  // world-space area of the hit triangle
    float cosLight  = abs(dot(N, rayDir));            // cosine at the light surface
    if (worldArea <= PDF_EPS || cosLight <= PDF_EPS || emissiveCount == 0u) // degenerate cases
        return 1.0;                                   // fall back to full weight (no MIS)
    bool  useAlias   = mesh.aliasAddress != 0 && mesh.area > 0.0; // is an alias table available for area-proportional picking?
    float selectProb = useAlias ? (objArea / mesh.area) : (1.0 / float(triCount)); // probability of picking this triangle
    float areaPdf    = selectProb / worldArea / float(emissiveCount); // area-measure pdf of sampling this point
    float pdfLight   = areaPdf * dist * dist / cosLight; // convert area pdf to solid-angle pdf
    return powerHeuristic(prevBrdfPdf, pdfLight);     // MIS weight for the BSDF strategy
}                                                     // end emitterMisWeight

public LightSample sampleEmissiveMesh(inout uint seed, InstanceGPU instance, MeshGPU mesh) { // pick a random point on an emissive mesh
    uint triCount = mesh.indexCount / 3;              // triangles in the mesh

    bool useAlias = mesh.aliasAddress != 0 && mesh.area > 0.0; // alias-table sampling (area-proportional) available?
    uint triIndex = min(uint(rand01(seed) * float(triCount)), triCount - 1u); // uniform triangle pick (also the alias bin)
    if (useAlias) {                                   // refine with the alias table?
        AliasEntry* aliasTable = (AliasEntry*)mesh.aliasAddress; // pointer to the alias table in device memory
        AliasEntry  entry      = aliasTable[triIndex]; // alias entry for this bin
        triIndex = (rand01(seed) < entry.prob) ? triIndex : entry.alias; // accept this bin or jump to its alias
    }                                                 // end alias refinement

    Vertex v0 = mesh.vertices[mesh.indices[triIndex * 3 + 0]]; // triangle vertex 0
    Vertex v1 = mesh.vertices[mesh.indices[triIndex * 3 + 1]]; // triangle vertex 1
    Vertex v2 = mesh.vertices[mesh.indices[triIndex * 3 + 2]]; // triangle vertex 2

    float r1 = sqrt(rand01(seed));                    // sqrt of a uniform -> uniform area sampling
    float r2 = rand01(seed);                          // second uniform
    float bu = 1.0 - r1;                              // barycentric for v0
    float bv = r1 * (1.0 - r2);                       // barycentric for v1
    float bw = r1 * r2;                               // barycentric for v2

    float4x4 xform = instance.model;                  // instance model matrix
    float3 lP = bu * v0.pos + bv * v1.pos + bw * v2.pos; // interpolated object-space position
    float3 lN = normalize(bu * v0.normal + bv * v1.normal + bw * v2.normal); // interpolated object-space normal

    lP = mul(xform, float4(lP, 1.0)).xyz;             // transform sample position to world space
    float3x3 nm = transpose(inverse3x3((float3x3)xform)); // normal matrix (inverse-transpose)
    lN = normalize(mul(nm, lN));                      // transform normal to world space

    float  objArea   = 0.5 * length(cross(v1.pos - v0.pos, v2.pos - v0.pos)); // object-space triangle area
    float3 e1        = mul(xform, float4(v1.pos - v0.pos, 0.0)).xyz; // world-space edge 1
    float3 e2        = mul(xform, float4(v2.pos - v0.pos, 0.0)).xyz; // world-space edge 2
    float  worldArea = 0.5 * length(cross(e1, e2));   // world-space triangle area

    LightSample ls;                                   // output sample
    ls.P        = lP;                                 // world position
    ls.N        = lN;                                 // world normal
    ls.uv       = bu * v0.texCoord + bv * v1.texCoord + bw * v2.texCoord; // interpolated UV
    ls.emission = float3(0.0);                        // textured emission applied by the caller (texture access lives in the scene module)
    float selectProb = useAlias ? (objArea / mesh.area) : (1.0 / float(triCount)); // probability this triangle was chosen
    ls.pdf      = (worldArea > PDF_EPS) ? selectProb / worldArea : 0.0; // area-measure pdf of the sample
    return ls;                                        // emissive sample
}                                                     // end sampleEmissiveMesh
