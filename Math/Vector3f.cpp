#include <cmath>
#include <optional>
#include "Vector3f.h"

__host__ __device__ void Vector3f::setX( float _x ) {
    x = _x;
}

__host__ __device__ void Vector3f::setY( float _y ) {
    y = _y;
}

__host__ __device__ void Vector3f::setZ( float _z ) {
    z = _z;
}

__host__ __device__ float Vector3f::getX() const {
    return x;
}

__host__ __device__ float Vector3f::getY() const {
    return y;
}

__host__ __device__ float Vector3f::getZ() const {
    return z;
}

__host__ __device__ void Vector3f::set( const Vector3f& p ) {
    x = p.x;
    y = p.y;
    z = p.z;
}

__host__ __device__ Vector3f Vector3f::normalize() const {
    float len = sqrt( pow( x, 2 ) +  pow( y, 2 ) + pow( z, 2 ));
    if ( len == 0 ) return *this;
    Vector3f res;
    res = *this;
    return res / len;
}

__host__ __device__ Vector3f Vector3f::cross( const Vector3f& vec ) const {
    return {
            y * vec.z - z * vec.y,
            z * vec.x - x * vec.z,
            x * vec.y - y * vec.x
    };
}

__host__ __device__ float& Vector3f::operator[]( int index ) {
    if ( index == 0 ) return x;
    if ( index == 1 ) return y;
    if ( index == 2 ) return z;
    return (float &) std::nullopt;
}

__host__ __device__ const float& Vector3f::operator[]( int index ) const {
    if ( index == 0 ) return x;
    if ( index == 1 ) return y;
    if ( index == 2 ) return z;
    return (float &) std::nullopt;
}

__host__ __device__ Vector3f& Vector3f::operator=( const Vector3f& p ) {
    set(p);
    return *this;
}

__host__ __device__ Vector3f Vector3f::operator+( const Vector3f& p ) const {
    return { x + p.x, y + p.y, z + p.z};
}
__host__ __device__ Vector3f Vector3f::operator-( const Vector3f& p ) const {
    return { x - p.x, y - p.y, z - p.z };
}

__host__ __device__ Vector3f Vector3f::operator*( float a ) const {
    return {x * a, y * a, z * a };
}

__host__ __device__ Vector3f Vector3f::operator/( float a ) const {
    return {x / a, y / a, z / a };
}

__host__ __device__ bool Vector3f::operator==( const Vector3f& p ) const {
    return ( x == p.x && y == p.y && z == p.z );
}

__host__ __device__ bool Vector3f::operator!=( const Vector3f& p ) const {
    return (!(*this == p));
}

__host__ __device__ Vector3f::Vector3f( const Vector3f& p ) {
    set(p);
}

__host__ __device__ Vector3f::Vector3f(): x( 0 ), y( 0 ), z( 0 ){ }

__host__ __device__ Vector3f::Vector3f(float _x, float _y, float _z): x( _x ), y( _y ), z( _z ){ }

__host__ __device__ Vector3f::~Vector3f() {
    x = 0;
    y = 0;
    z = 0;
}