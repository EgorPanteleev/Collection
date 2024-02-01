#pragma once
#include "Vector4f.h"
class Mat4f {
public:
    Mat4f();
    Mat4f( const Vector4f& vec1, const Vector4f& vec2, const Vector4f& vec3, const Vector4f& vec4 );
    Vector4f& operator[]( int index );
    const Vector4f& operator[]( int index ) const;
    [[nodiscard]] float getDet() const;
    [[nodiscard]] Mat4f transpose() const;
    [[nodiscard]] Mat4f inverse() const;
    [[nodiscard]] static Mat4f identity();
private:
    Vector4f columns[4];
};

