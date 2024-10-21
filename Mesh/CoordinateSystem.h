//
// Created by auser on 8/27/24.
//

#ifndef COLLECTION_COORDINATESYSTEM_H
#define COLLECTION_COORDINATESYSTEM_H
#include "Vec3.h"
#include "Mat3.h"

class CoordinateSystem {
public:
    CoordinateSystem();
    CoordinateSystem( const Vec3d& N );
    Mat3d getOrthonormalBasis( const Vec3d& N ) const;
    Vec3d getNormal() const;
    Vec3d from( const Vec3d& vec ) const;
    Vec3d to( const Vec3d& vec ) const;
protected:
    Mat3d T;
};


#endif //COLLECTION_COORDINATESYSTEM_H
