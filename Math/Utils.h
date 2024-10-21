//
// Created by auser on 10/20/24.
//

#ifndef COLLECTION_UTILS_H
#define COLLECTION_UTILS_H
#include "RGB.h"
#include "Vec3.h"

double saturate( double z );

double pow2( double a );

template <typename Type>
Vec3<Type> toNormal( const RGB& col ) {
    Vec3<Type> res = { col.r, col.g, col.b };
    res = res / 255.0 * 2.0 - 1;
    return res.normalize();
}

template<typename Type>
Vec3<Type> toVec3( const RGB& col ) {
    return { (Type) col.r, (Type) col.g, (Type) col.b };
}
template<typename Type>
RGB toRGB( const Vec3<Type>& vec ) {
    return { (double) vec[0], (double) vec[1], (double) vec[2] };
}

#endif //COLLECTION_UTILS_H
