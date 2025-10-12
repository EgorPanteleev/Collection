//
// Created by auser on 8/1/24.
//

#include "BBox.h"

BBox::BBox(): pMin( 1e30 ), pMax( -1e30 ) {}

BBox::BBox( const Vec3d& pMin, const Vec3d& pMax ): pMin( pMin ), pMax( pMax ) {}

void BBox::merge( const BBox &bbox ) {
//    if (bbox.pMin[0] == 1e30) return;
    merge( bbox.pMin );
    merge( bbox.pMax );
}

void BBox::merge(const Vec3d &p) {
    pMin = min( p, pMin );
    pMax = max( p, pMax );
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