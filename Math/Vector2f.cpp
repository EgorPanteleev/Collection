
//
// Created by igor on 28.01.2024.
//

#include "Vector2f.h"
#include <cmath>
#include <optional>
__host__ __device__ void Vector2f::setX( float _x ) {
    x = _x;
}

__host__ __device__ void Vector2f::setY( float _y ) {
    y = _y;
}

__host__ __device__ float Vector2f::getX() const {
    return x;
}

__host__ __device__ float Vector2f::getY() const {
    return y;
}

__host__ __device__ void Vector2f::set( const Vector2f& p ) {
    x = p.getX();
    y = p.getY();
}

__host__ __device__ Vector2f Vector2f::normalize() const {
    Vector2f res;
    res = *this;
    float len = sqrt( pow( x, 2 ) +  pow( y, 2 ));
    return res/len;
}

__host__ __device__ float& Vector2f::operator[]( int index ) {
    if ( index == 0) return x;
    if ( index == 1) return y;
    return (float &) std::nullopt;
}

__host__ __device__ const float& Vector2f::operator[]( int index ) const {
    if ( index == 0) return x;
    if ( index == 1) return y;
    return (float &) std::nullopt;
}

__host__ __device__ Vector2f& Vector2f::operator=( const Vector2f& p ) {
    set(p);
    return *this;
}

__host__ __device__ Vector2f Vector2f::operator+( const Vector2f& p ) const {
    Vector2f ret;
    ret.setX( x + p.getX() );
    ret.setY( y + p.getY() );
    return ret;
}

__host__ __device__ Vector2f Vector2f::operator-( const Vector2f& p ) const {
    Vector2f ret;
    ret.setX( x - p.getX() );
    ret.setY( y - p.getY() );
    return ret;
}

__host__ __device__ Vector2f Vector2f::operator*( float a ) const {
    Vector2f ret;
    ret.setX( x * a );
    ret.setY( y * a );
    return ret;
}

__host__ __device__ Vector2f Vector2f::operator/( float a ) const {
    Vector2f ret;
    ret.setX( x / a );
    ret.setY( y / a );
    return ret;
}

__host__ __device__ bool Vector2f::operator==( const Vector2f& p ) const {
    return ( x == p.getX() && y == p.getY() );
}

__host__ __device__ bool Vector2f::operator!=( const Vector2f& p ) const {
    return (!(*this == p));
}

__host__ __device__ Vector2f::Vector2f( const Vector2f& p ) {
    set(p);
}

__host__ __device__ Vector2f::Vector2f(): x( 0 ), y( 0 ) { }

__host__ __device__ Vector2f::Vector2f(float _x, float _y ): x( _x ), y( _y ){ }

__host__ __device__ Vector2f::~Vector2f() {
    x = 0;
    y = 0;
}