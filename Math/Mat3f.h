#pragma once
#include "Vector3f.h"
class Mat3f {
public:
    Mat3f();
    Mat3f( const Vector3f& vec1, const Vector3f& vec2, const Vector3f& vec3 );
    Vector3f& operator[]( int index );
    const Vector3f& operator[]( int index ) const;
    bool operator==( Mat3f& mat ) const;
    bool operator!=( Mat3f& mat ) const;
    [[nodiscard]] float getAlgExtension( int col, int row ) const;
    [[nodiscard]] float getDet() const;
    [[nodiscard]] Mat3f transpose() const;
    [[nodiscard]] Mat3f inverse() const;
    [[nodiscard]] Mat3f getUnion() const;
    [[nodiscard]] static Mat3f identity();
private:
    Vector3f columns[3];
};

