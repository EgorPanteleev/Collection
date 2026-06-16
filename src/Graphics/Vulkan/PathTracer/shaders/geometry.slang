module geometry;

import common;
import types;

public struct Triangle {
    public Vertex v0;
    public Vertex v1;
    public Vertex v2;
};

public struct HitGeometry {
    public float3 P;
    public float3 N;
    public float3 gN;
    public float2 uv;
};

public Triangle fetchTriangle(MeshGPU mesh, uint primIndex) {
    Triangle tri;
    tri.v0 = mesh.vertices[mesh.indices[primIndex * 3 + 0]];
    tri.v1 = mesh.vertices[mesh.indices[primIndex * 3 + 1]];
    tri.v2 = mesh.vertices[mesh.indices[primIndex * 3 + 2]];
    return tri;
}

public HitGeometry interpolateHit(Triangle tri, float2 bary, float3x4 o2w, float3 rayDir) {
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;

    float3x3 normalMatrix = transpose(inverse3x3((float3x3)o2w));

    HitGeometry hit;
    hit.uv = w * tri.v0.texCoord + u * tri.v1.texCoord + v * tri.v2.texCoord;
    hit.P  = mul(o2w, float4(w * tri.v0.pos + u * tri.v1.pos + v * tri.v2.pos, 1.0));
    hit.N  = normalize(mul(normalMatrix, w * tri.v0.normal + u * tri.v1.normal + v * tri.v2.normal));
    if (dot(hit.N, rayDir) > 0.0) hit.N = -hit.N;

    float3 gN = normalize(cross(tri.v1.pos - tri.v0.pos, tri.v2.pos - tri.v0.pos));
    gN = normalize(mul(normalMatrix, gN));
    if (dot(gN, hit.N) < 0.0) gN = -gN;
    hit.gN = gN;
    return hit;
}
