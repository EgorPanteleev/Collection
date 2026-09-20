module types;

#include "SharedTypes.h"

public struct PathPayload {
    public float3 throughput;
    public float3 radiance;
    public float3 origin;
    public float3 direction;
    public uint   instanceId;
    public uint   seed;
    public float  prevBrdfPdf;
    public float  segmentDist;
    public bool   done;
    public float3 mediumAbsorption;
};

public struct ShadowPayload {
    public float visible;
    public uint  seed;
};
