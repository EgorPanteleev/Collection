//
// Created by auser on 8/27/24.
//

#ifndef COLLECTION_COORDINATESYSTEM_H
#define COLLECTION_COORDINATESYSTEM_H
#include "Vec3.h"
#include "Mat3.h"
#include "SystemUtils.h"

class CoordinateSystem {
public:
    CoordinateSystem() = delete;
    HOST_DEVICE CoordinateSystem( const Vec3d& N ): T( getOrthonormalBasis( N ) ) {
    }
    HOST_DEVICE Mat3d getOrthonormalBasis( const Vec3d& N ) const {
        double sign = std::copysign(1.0, N[2]);
        double a = -1.0 / (sign + N[2]);
        double b = N[0] * N[1] * a;
        return { { 1.0 + sign * N[0] * N[0] * a, sign * b, -sign * N[0] },
                 { b, sign + N[1] * N[1] * a, -N[1]                      },
                 N                                                     };
    }

    HOST_DEVICE Vec3d getNormal() const {
        return T[2];
    }
    HOST_DEVICE Vec3d from( const Vec3d& vec ) const {
        return T * vec;
    }

    HOST_DEVICE Vec3d to( const Vec3d& vec ) const {
        return T.transpose() * vec;
    }
protected:
    Mat3d T;
};


#endif //COLLECTION_COORDINATESYSTEM_H
