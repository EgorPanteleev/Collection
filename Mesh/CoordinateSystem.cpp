//
// Created by auser on 8/27/24.
//

#include "CoordinateSystem.h"
#include <cmath>

CoordinateSystem::CoordinateSystem(): T() {
}

CoordinateSystem::CoordinateSystem( const Vec3d& N ): T( getOrthonormalBasis( N ) ) {
}

Mat3d CoordinateSystem::getOrthonormalBasis( const Vec3d& N ) const {
    double sign = std::copysign(1.0, N[2]);
    double a = -1.0 / (sign + N[2]);
    double b = N[0] * N[1] * a;
    return { { 1.0 + sign * N[0] * N[0] * a, sign * b, -sign * N[0] },
             { b, sign + N[1] * N[1] * a, -N[1]                      },
               N                                                     };
}

Vec3d CoordinateSystem::getNormal() const {
    return T[2];
}

Vec3d CoordinateSystem::from( const Vec3d& vec ) const {
    return T * vec;
}

Vec3d CoordinateSystem::to( const Vec3d& vec ) const {
    return T.transpose() * vec;
}