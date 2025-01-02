//
// Created by auser on 8/1/24.
//

#ifndef COLLECTION_BBOX_H
#define COLLECTION_BBOX_H
#include <cmath>
#include "Vec3.h"
#include "Ray.h"

class BBox {
public:
    BBox();

    BBox( const Vec3d& pMin, const Vec3d& pMax );

    [[nodiscard]] Vec3d getCentroid() const;

    void merge( const BBox &bbox );

    void merge(const Vec3d &p);

    [[nodiscard]] HOST_DEVICE bool intersectsWithRay( const Ray& ray ) const {
        Vec3d invDirection = 1.0 / ray.direction;
        Vec3d t0( (pMin - ray.origin) * invDirection );
        Vec3d t1( (pMax - ray.origin) * invDirection );
        double tmin = std::min( t0[0], t1[0] );
        double tmax = std::max( t0[0], t1[0] );
        tmin = std::max( tmin, std::min( t0[1], t1[1] ) );
        tmax = std::min( tmax, std::max( t0[1], t1[1] ) );
        tmin = std::max( tmin, std::min( t0[2], t1[2] ) );
        tmax = std::min( tmax, std::max( t0[2], t1[2] ) );
        return tmax >= tmin && tmin < 1e30f && tmax > 0;
    }

    [[nodiscard]] double getArea() const;

    bool operator==( const BBox& bbox ) const;

    bool operator!=( const BBox& bbox ) const;
public:
    Vec3d pMin;
    Vec3d pMax;
};

HOST_DEVICE inline std::ostream& operator << (std::ostream &os, const BBox &bbox ) {
    return os << bbox.pMin << bbox.pMax;
}

#endif //COLLECTION_BBOX_H
