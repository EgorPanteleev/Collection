//
// Created by auser on 10/20/24.
//

#ifndef COLLECTION_UTILS_H
#define COLLECTION_UTILS_H
#include "RGB.h"
#include "Vec3.h"

template <typename Type>
inline Type saturate( Type z ) {
    if ( z < 0.0f ) return 0.0;
    if ( z > 1.0f ) return 1.0;
    return z;
}

template <typename Type>
inline Type pow2( Type a ) {
    return a * a;
}

template <typename Type>
inline Vec3<Type> toNormal( const RGB& col ) {
    Vec3<Type> res = { col.r, col.g, col.b };
    res = res / 255.0 * 2.0 - 1;
    return res.normalize();
}

template<typename Type>
inline Vec3<Type> toVec3( const RGB& col ) {
    return { (Type) col.r, (Type) col.g, (Type) col.b };
}
template<typename Type>
inline RGB toRGB( const Vec3<Type>& vec ) {
    return { (double) vec[0], (double) vec[1], (double) vec[2] };
}

#endif //COLLECTION_UTILS_H
