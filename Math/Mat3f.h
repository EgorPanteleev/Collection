#pragma once
#include "Vector3f.h"
class Mat3f {
public:
    Mat3f();
    Mat3f( Vector3f vec1, Vector3f vec2, Vector3f vec3 );
    Vector3f& operator[]( int index );
    const Vector3f& operator[]( int index ) const;
    float getDet() const;
    Mat3f transpose() const;
    Mat3f inverse() const;
private:
    Vector3f columns[3];
};

