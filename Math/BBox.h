//
// Created by auser on 8/1/24.
//

#ifndef COLLECTION_BBOX_H
#define COLLECTION_BBOX_H
#include "Vector3f.h"
#include "Utils.h"
#include <cmath>

struct BBox {
    BBox(): pMin( { 1e30f, 1e30f , 1e30f } ), pMax( { -1e30f, -1e30f , -1e30f } ) {}
    BBox( const Vector3f& pMin, const Vector3f& pMax ): pMin( pMin ), pMax( pMax ) {}
    Vector3f centroid() {
        return (pMax + pMin) / 2.0;
    }
    void merge( const BBox &bbox ) {
        if (bbox.pMin.x == 1e30f) return;
        merge( bbox.pMin );
        merge( bbox.pMax );
    }

    void merge(const Vector3f &p) {
        pMin = min( p, pMin );
        pMax = max( p, pMax );
    }
    float area() {
        Vector3f d = pMax - pMin;
        return (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    bool intersectsWithRay( const Vector3f& origin, const Vector3f& direction ) const {
        Vector3f t0{ (pMin - origin) / direction };
        Vector3f t1{ (pMax - origin) / direction };
        float tmin = std::min( t0.x, t1.x );
        float tmax = std::max( t0.x, t1.x );
        tmin = std::max( tmin, std::min( t0.y, t1.y ) );
        tmax = std::min( tmax, std::max( t0.y, t1.y ) );
        tmin = std::max( tmin, std::min( t0.z, t1.z ) );
        tmax = std::min( tmax, std::max( t0.z, t1.z ) );
        return tmax >= tmin && tmin < 1e30f && tmax > 0;
    }

    Vector3f pMin;
    Vector3f pMax;
};

#endif //COLLECTION_BBOX_H
