#pragma once
#include "Vector4f.h"
class Mat4f {
public:
    Mat4f();
    Mat4f( Vector4f vec1, Vector4f vec2, Vector4f vec3, Vector4f vec4 );
    Vector4f& operator[]( int index );
    const Vector4f& operator[]( int index ) const;
    float getDet() const;
    Mat4f transpose() const;
    Mat4f inverse() const;
private:
    Vector4f columns[4];
};

