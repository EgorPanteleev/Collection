module types;

public struct Vertex {
    public float3 pos;
    public float2 texCoord;
    public float3 n;
    public float4 tangent;
};

public struct AliasEntry {
    public float prob;
    public uint  alias;
};

public struct MeshInfo {
    public Vertex*  vertices;
    public uint*    indices;
    public uint     indexCount;
    public float    area;
    public uint64_t aliasAddress;
};

public struct InstanceData {
    public float4x4 model;
    public uint     meshIndex;
    public uint     materialIndex;
};

public struct MaterialData {
    public float3 baseColor;
    public float3 emissionColor;
    public float  luminance;
    public uint   baseColorTexIndex;
    public uint   normalTexIndex;
};

public struct PathPayload {
    public float3 throughput;
    public float3 radiance;
    public float3 origin;
    public float3 direction;
    public uint   instanceId;
    public uint   seed;
    public uint   depth;
    public float  prevBrdfPdf;
    public bool   done;
};
