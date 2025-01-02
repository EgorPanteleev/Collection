//
// Created by auser on 8/1/24.
//

#include "BBox.h"

BBox::BBox(): pMin( { 1e30, 1e30 , 1e30 } ), pMax( { -1e30, -1e30 , -1e30 } ) {}

BBox::BBox( const Vec3d& pMin, const Vec3d& pMax ): pMin( pMin ), pMax( pMax ) {}

Vec3d BBox::getCentroid() const {
    return (pMax + pMin) / 2.0;
}

void BBox::merge( const BBox &bbox ) {
    if (bbox.pMin[0] == 1e30) return;
    merge( bbox.pMin );
    merge( bbox.pMax );
}

void BBox::merge(const Vec3d &p) {
    pMin = min( p, pMin );
    pMax = max( p, pMax );
}

bool BBox::intersectsWithRay( const Vec3d& origin, const Vec3d& direction ) const {
    Vec3d t0{ (pMin - origin) / direction };
    Vec3d t1{ (pMax - origin) / direction };
    double tmin = std::min( t0[0], t1[0] );
    double tmax = std::max( t0[0], t1[0] );
    tmin = std::max( tmin, std::min( t0[1], t1[1] ) );
    tmax = std::min( tmax, std::max( t0[1], t1[1] ) );
    tmin = std::max( tmin, std::min( t0[2], t1[2] ) );
    tmax = std::min( tmax, std::max( t0[2], t1[2] ) );
    return tmax >= tmin && tmin < 1e30f && tmax > 0;
}

double BBox::getArea() const {
    Vec3d d = pMax - pMin;
    return (d[0] * d[1] + d[0] * d[2] + d[1] * d[2]);
}

bool BBox::operator==( const BBox& bbox ) const {
    return ( pMin == bbox.pMin && pMax == bbox.pMax );
}

bool BBox::operator!=( const BBox& bbox ) const {
    return ( pMin != bbox.pMin || pMax != bbox.pMax );
}