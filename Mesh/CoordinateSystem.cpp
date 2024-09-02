//
// Created by auser on 8/27/24.
//

#include "CoordinateSystem.h"
#include <cmath>
#include "Utils.h"

CoordinateSystem::CoordinateSystem(): T() {
}

CoordinateSystem::CoordinateSystem( const Vector3f& N ): T( getOrthonormalBasis( N ) ) {
}

Mat3f CoordinateSystem::getOrthonormalBasis( const Vector3f& N ) const {
    float sign = std::copysign(1.0f, N.z);
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    return { { 1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x },
             { b, sign + N.y * N.y * a, -N.y                      },
               N                                                  };
}

Vector3f CoordinateSystem::getNormal() const {
    return T[2];
}

Vector3f CoordinateSystem::from( const Vector3f& vec ) const {
    return T * vec;
}

Vector3f CoordinateSystem::to( const Vector3f& vec ) const {
    return T.transpose() * vec;
}