//
// Created by auser on 11/26/24.
//
#ifndef COLLECTION_REFACTOR_UTILS_H
#define COLLECTION_REFACTOR_UTILS_H

#include <limits>
#include "Random.h"
#include "Vec3.h"
#include "OutputStream.h"

constexpr double EPS = 0.0001;

constexpr double INF = std::numeric_limits<double>::infinity();

extern OutputStream<std::ostream> MESSAGE;

//HOST_DEVICE double clamp( double a, double border1, double border2 ) {
//    if ( a < border1 ) return border1;
//    if ( a > border2 ) return border2;
//    return a;
//}

template <typename Type>
inline HOST_DEVICE Type saturate( Type z ) {
    if ( z < 0.0f ) return 0.0;
    if ( z > 1.0f ) return 1.0;
    return z;
}


template <typename Type>
inline HOST_DEVICE Type pow2( Type a ) {
    return a * a;
}

template <typename Type>
inline Type toRadians( Type degrees ) {
    return degrees * ( M_PI / 180 );
}

template <typename Type>
inline Type toDegrees( Type radians ) {
    return radians * ( 180 / M_PI );
}

#endif //COLLECTION_REFACTOR_UTILS_H

