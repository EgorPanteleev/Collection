#include "Vector4f.h"
#include <cmath>
#include "Utils.h"
void Vector4f::setX( float _x ) {
    x = _x;
}

void Vector4f::setY( float _y ) {
    y = _y;
}

void Vector4f::setZ( float _z ) {
    z = _z;
}

void Vector4f::setW( float _w ) {
    w = _w;
}

float Vector4f::getX() const {
    return x;
}

float Vector4f::getY() const {
    return y;
}

float Vector4f::getZ() const {
    return z;
}

float Vector4f::getW() const {
    return w;
}

void Vector4f::set( const Vector4f& p ) {
    x = p.getX();
    y = p.getY();
    z = p.getZ();
    w = p.getW();
}

Vector4f Vector4f::normalize() {
    float lenght = sqrt( pow( x, 2 ) +  pow( y, 2 ) + pow( z, 2 ) + pow( w, 2));
    return *this/lenght;
}

Vector4f Vector4f::cross( Vector4f vec ) {
    Vector3f vec1 = Vector3f(x, y, z);
    Vector3f vec2 = Vector3f(vec[0], vec[1], vec[2]);
    Vector3f res = vec1.cross(vec2);
    return Vector4f(res[0], res[1], res[2], 1);
}

float Vector4f::operator[]( int index ) {
    if ( index == 0) return x;
    if ( index == 1) return y;
    if ( index == 2) return z;
    if ( index == 3) return w;
}

const float Vector4f::operator[]( int index ) const {
    if ( index == 0) return x;
    if ( index == 1) return y;
    if ( index == 2) return z;
    if ( index == 3) return w;
}

void Vector4f::operator=( const Vector4f& p ) {
    set(p);
}

Vector4f Vector4f::operator+( const Vector4f& p ) const {
    Vector4f ret;
    ret.setX( x + p.getX() );
    ret.setY( y + p.getY() );
    ret.setZ( z + p.getZ() );
    ret.setZ( w + p.getW() );
    return ret;
}

Vector4f Vector4f::operator-( const Vector4f& p ) const {
    Vector4f ret;
    ret.setX( x - p.getX() );
    ret.setY( y - p.getY() );
    ret.setZ( z - p.getZ() );
    ret.setZ( w - p.getW() );
    return ret;
}

Vector4f Vector4f::operator*( float a ) const {
    Vector4f ret;
    ret.setX( x * a );
    ret.setY( y * a );
    ret.setZ( z * a );
    ret.setZ( w * a );
    return ret;
}

Vector4f Vector4f::operator/( float a ) const {
    Vector4f ret;
    ret.setX( x / a );
    ret.setY( y / a );
    ret.setZ( z / a );
    ret.setZ( w / a );
    return ret;
}

bool Vector4f::operator==( const Vector4f& p ) {
    return ( x == p.getX() && y == p.getY() && z == p.getZ() && w == p.getW() );
}

bool Vector4f::operator!=( const Vector4f& p ) {
    return (!(*this == p));
}

Vector4f::Vector4f( const Vector4f& p ) {
    set(p);
}

Vector4f::Vector4f(): x( 0 ), y( 0 ), z( 0 ), w( 0 ){ }

Vector4f::Vector4f(float _x, float _y, float _z, float _w): x( _x ), y( _y ), z( _z ), w( _w ){ }

Vector4f::~Vector4f() {
    x = 0;
    y = 0;
    z = 0;
    w = 0;
}