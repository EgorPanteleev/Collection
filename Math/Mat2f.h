#pragma once
#include "Vector2f.h"

class Mat2f {
public:
    Mat2f();
    Mat2f( const Vector2f& vec1, const Vector2f& vec2 );
    Vector2f& operator[]( int index );
    const Vector2f& operator[]( int index ) const;
    bool operator==( Mat2f& mat ) const;
    bool operator!=( Mat2f& mat ) const;
    [[nodiscard]] float getDet() const;
    [[nodiscard]] Mat2f transpose() const;
    [[nodiscard]] Mat2f inverse() const;
    [[nodiscard]] Mat2f getUnion() const;
private:
    Vector2f columns[3];
};


