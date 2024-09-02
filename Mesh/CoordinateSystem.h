//
// Created by auser on 8/27/24.
//

#ifndef COLLECTION_COORDINATESYSTEM_H
#define COLLECTION_COORDINATESYSTEM_H
#include "Vector3f.h"
#include "Mat3f.h"

class CoordinateSystem {
public:
    CoordinateSystem();
    CoordinateSystem( const Vector3f& N );
    Mat3f getOrthonormalBasis( const Vector3f& N ) const;
    Vector3f getNormal() const;
    Vector3f from( const Vector3f& vec ) const;
    Vector3f to( const Vector3f& vec ) const;
protected:
    Mat3f T;
};


#endif //COLLECTION_COORDINATESYSTEM_H
