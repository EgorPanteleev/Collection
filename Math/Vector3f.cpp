#include "Vector3f.h"
#include <cmath>
#include <optional>
void Vector3f::setX( float _x ) {
    x = _x;
}

void Vector3f::setY( float _y ) {
    y = _y;
}

void Vector3f::setZ( float _z ) {
    z = _z;
}

float Vector3f::getX() const {
    return x;
}

float Vector3f::getY() const {
    return y;
}

float Vector3f::getZ() const {
    return z;
}

void Vector3f::set( const Vector3f& p ) {
    x = p.x;
    y = p.y;
    z = p.z;
}

Vector3f Vector3f::normalize() const {
    float len = sqrt( pow( x, 2 ) +  pow( y, 2 ) + pow( z, 2 ));
    if ( len == 0 ) return *this;
    Vector3f res;
    res = *this;
    return res / len;
}

Vector3f Vector3f::cross( const Vector3f& vec ) const {
    return {
            y * vec.z - z * vec.y,
            z * vec.x - x * vec.z,
            x * vec.y - y * vec.x
    };
}

float& Vector3f::operator[]( int index ) {
    if ( index == 0 ) return x;
    if ( index == 1 ) return y;
    if ( index == 2 ) return z;
    return (float &) std::nullopt;
}

const float& Vector3f::operator[]( int index ) const {
    if ( index == 0 ) return x;
    if ( index == 1 ) return y;
    if ( index == 2 ) return z;
    return (float &) std::nullopt;
}

Vector3f& Vector3f::operator=( const Vector3f& p ) {
    set(p);
    return *this;
}

Vector3f Vector3f::operator+( const Vector3f& p ) const {
    return { x + p.x, y + p.y, z + p.z};
}
Vector3f Vector3f::operator-( const Vector3f& p ) const {
    return { x - p.x, y - p.y, z - p.z };
}

Vector3f Vector3f::operator-( float a ) const {
    return { x - a, y - a, z - a };
}

Vector3f Vector3f::operator+( float a ) const {
    return { x + a, y + a, z + a };
}

Vector3f Vector3f::operator*( float a ) const {
    return {x * a, y * a, z * a };
}

Vector3f Vector3f::operator*( const Vector3f& p ) const {
    return {x * p.x, y * p.y, z * p.z };
}

Vector3f Vector3f::operator/( const Vector3f& p ) const {
    return {x / p.x, y / p.y, z / p.z };
}

Vector3f Vector3f::operator/( float a ) const {
    return {x / a, y / a, z / a };
}

bool Vector3f::operator==( const Vector3f& p ) const {
    return ( x == p.x && y == p.y && z == p.z );
}

bool Vector3f::operator!=( const Vector3f& p ) const {
    return (!(*this == p));
}

Vector3f::Vector3f( const Vector3f& p ) {
    set(p);
}

Vector3f::Vector3f(): x( 0 ), y( 0 ), z( 0 ){ }

Vector3f::Vector3f(float _x, float _y, float _z): x( _x ), y( _y ), z( _z ){ }

Vector3f::~Vector3f() {
    x = 0;
    y = 0;
    z = 0;
}

std::ostream& operator << (std::ostream &os, const Vector3f &v ) {
    return os << "( " << v.x << ", " << v.y << ", " << v.z << " )";
}