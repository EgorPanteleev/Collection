//
// Created by auser on 5/12/24.
//

#ifndef COLLECTION_INTERSECTIONDATA_H
#define COLLECTION_INTERSECTIONDATA_H

#include "Vector3f.h"
#include "Primitive.h"
#include "CoordinateSystem.h"
class Triangle;
class Sphere;

struct TraceData {
    TraceData(): primitive( nullptr ), cs(), material(), P(), t( __FLT_MAX__ ), ambientOcclusion() {}
    void load () {
        cs = { primitive->getNormal( P ) };
        material.setColor( primitive->getColor( P ) );
        material.setIntensity( primitive->getMaterial().getIntensity() );
        material.setRoughness( primitive->getRoughness( P ) );
        material.setMetalness( primitive->getMetalness( P ) );
        ambientOcclusion = primitive->getAmbient( P ).r;
    }
    [[nodiscard]] RGB getColor() const {
        return material.getColor();
    }
    [[nodiscard]] float getIntensity() const {
        return material.getIntensity();
    }
    [[nodiscard]] float getRoughness() const {
        return material.getRoughness();
    }
    [[nodiscard]] float getMetalness() const {
        return material.getMetalness();
    }
    Primitive* primitive;
    Vector3f P;
    float t;
    CoordinateSystem cs;
    Material material;
    float ambientOcclusion;
};

class IntersectionData {
public:
    IntersectionData();
    IntersectionData( float t, Primitive* prim );
    float t;
    Primitive* primitive;
};

#endif //COLLECTION_INTERSECTIONDATA_H
