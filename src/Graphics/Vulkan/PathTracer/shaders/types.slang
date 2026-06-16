module types;

#include "SharedTypes.h"

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
