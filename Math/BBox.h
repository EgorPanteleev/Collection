//
// Created by auser on 8/1/24.
//

#ifndef COLLECTION_BBOX_H
#define COLLECTION_BBOX_H
#include "Vec3.h"
#include <cmath>

class BBox {
public:
    BBox();

    BBox( const Vec3d& pMin, const Vec3d& pMax );

    Vec3d getCentroid() const;

    void merge( const BBox &bbox );

    void merge(const Vec3d &p);

    bool intersectsWithRay( const Vec3d& origin, const Vec3d& direction ) const;

    double getArea() const;

    bool operator==( const BBox& bbox ) const;

    bool operator!=( const BBox& bbox ) const;
public:
    Vec3d pMin;
    Vec3d pMax;
};

#endif //COLLECTION_BBOX_H
