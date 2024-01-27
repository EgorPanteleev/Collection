#pragma once
#include "Vector3f.h"
class Mat3f {
public:
    Mat3f();
    Mat3f( const Vector3f& vec1, const Vector3f& vec2, const Vector3f& vec3 );
    Vector3f& operator[]( int index );
    const Vector3f& operator[]( int index ) const;
    [[nodiscard]] float getDet() const;
    [[nodiscard]] Mat3f transpose() const;
    [[nodiscard]] Mat3f inverse() const;
private:
    Vector3f columns[3];
};

